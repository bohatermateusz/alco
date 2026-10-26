package com.hismith.hismithapp;

import android.content.Context;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.text.Editable;
import android.text.TextUtils;
import android.text.TextWatcher;
import android.view.View;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.TextView;
import com.facebook.internal.AnalyticsEvents;
import com.facebook.internal.NativeProtocol;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.http.RetrofitRequest;
import com.hismith.hismithapp.http.RxSubscribe;
import com.hismith.hismithapp.manager.PictureResultManager;
import com.hismith.hismithapp.model.TopicSendModel;
import com.hismith.hismithapp.model.UploadModel;
import com.hismith.hismithapp.picker.PictureSelectorManager;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.RequestBodyUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import com.hismith.hismithapp.utils.SoftKeyboardUtil;
import com.hismith.hismithapp.utils.StringUtils;
import com.hismith.hismithapp.utils.ToastUtil;
import com.hismith.hismithapp.view.PictureAndTextEditorView;
import com.luck.picture.lib.entity.LocalMedia;
import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.disposables.CompositeDisposable;
import io.reactivex.schedulers.Schedulers;
import java.io.File;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import okhttp3.RequestBody;

/* loaded from: classes3.dex */
public class PublishActivity extends BaseActivity implements View.OnClickListener, TextWatcher, View.OnFocusChangeListener {
    public static final String EXTRA_POST_ID = "extra_post_id";
    private static final int WHAT_PUBLISH = 1;
    private RetrofitRequest baseRequest;
    private RetrofitRequest fileRequest;
    private String mCategorieId;
    private CompositeDisposable mCompositeDisposable;
    private PictureAndTextEditorView mEditContent;
    private List<View> mEditList;
    private EditText mEditTitle;
    private ImageView mImgAlbum;
    private ImageView mImgCamera;
    private LinearLayout mLayoutAction;
    private LinearLayout mLayoutEdit;
    private ProgressBar mProgress;
    private ImageView mTitleBack;
    private ProgressBar mTitleProgress;
    private TextView mTxtComplete;
    private TextView mTxtPublish;
    private PictureResultManager pictureResultManager;
    private Thread publishThread;
    private SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
    private Map<String, String> replacePathMap = new HashMap();
    private final int REQUEST_CAMERA = 777;
    private final int REQUEST_ALBUM = 888;
    private Handler mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.PublishActivity.1
        @Override // android.os.Handler.Callback
        public boolean handleMessage(Message message) {
            if (message.what != 1) {
                return false;
            }
            PublishActivity.this.topicSend();
            return false;
        }
    });
    private boolean isImage = false;
    private int startIndex = -1;
    private int endIndex = -1;
    private int mTagCount = 0;

    public static Intent getAppDetailSettingIntent(Context context) {
        Intent intent = new Intent();
        intent.addFlags(268435456);
        intent.setAction("android.settings.APPLICATION_DETAILS_SETTINGS");
        intent.setData(Uri.fromParts("package", context.getPackageName(), null));
        return intent;
    }

    private void init() {
        if (getIntent() != null) {
            this.mCategorieId = getIntent().getStringExtra(EXTRA_POST_ID);
        }
        this.mCompositeDisposable = new CompositeDisposable();
        this.baseRequest = new RetrofitRequest();
        this.fileRequest = new RetrofitRequest(RetrofitRequest.EnumRequest.REQUEST_FILE);
        this.pictureResultManager = new PictureResultManager(this);
    }

    private void initView() {
        this.mTitleBack = (ImageView) findViewById(R.id.title_img_back);
        this.mTxtPublish = (TextView) findViewById(R.id.publish_txt_publish);
        this.mTitleProgress = (ProgressBar) findViewById(R.id.publish_title_progress);
        this.mEditTitle = (EditText) findViewById(R.id.publish_edit_title);
        this.mEditContent = (PictureAndTextEditorView) findViewById(R.id.publish_edit_content);
        this.mLayoutEdit = (LinearLayout) findViewById(R.id.publish_layout_edit);
        this.mLayoutAction = (LinearLayout) findViewById(R.id.publish_layout_action);
        this.mImgCamera = (ImageView) findViewById(R.id.publish_img_camera);
        this.mImgAlbum = (ImageView) findViewById(R.id.publish_img_album);
        this.mTxtComplete = (TextView) findViewById(R.id.publish_txt_complete);
        this.mProgress = (ProgressBar) findViewById(R.id.publish_progress);
        LanguageUtil.setText(this, this.mTxtPublish, "publish");
        LanguageUtil.setHint(this, this.mEditTitle, "enterTitle");
        LanguageUtil.setHint(this, this.mEditContent, "enterContent");
        LanguageUtil.setText(this, this.mImgCamera, "shoot");
        LanguageUtil.setText(this, this.mImgAlbum, AnalyticsEvents.PARAMETER_SHARE_DIALOG_CONTENT_PHOTO);
        LanguageUtil.setText(this, this.mTxtComplete, "publish");
        ArrayList arrayList = new ArrayList();
        this.mEditList = arrayList;
        arrayList.add(this.mEditTitle);
        this.mEditList.add(this.mEditContent);
    }

    private void setListener() {
        this.mTitleBack.setOnClickListener(this);
        this.mTxtPublish.setOnClickListener(this);
        this.mImgCamera.setOnClickListener(this);
        this.mImgAlbum.setOnClickListener(this);
        this.mEditContent.addTextChangedListener(this);
        this.mEditContent.setOnFocusChangeListener(this);
        this.mTxtComplete.setOnClickListener(this);
    }

    private void startSync() {
        stopSync();
        LogUtil.e("startSync()");
        Thread thread = new Thread() { // from class: com.hismith.hismithapp.PublishActivity.4
            @Override // java.lang.Thread, java.lang.Runnable
            public void run() {
                boolean z;
                do {
                    try {
                        Iterator it = PublishActivity.this.replacePathMap.keySet().iterator();
                        z = false;
                        while (it.hasNext()) {
                            if (TextUtils.isEmpty((CharSequence) PublishActivity.this.replacePathMap.get((String) it.next()))) {
                                z = true;
                            }
                        }
                        LogUtil.d("startSync() flag：" + z);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                } while (z);
                Thread.sleep(500L);
                PublishActivity.this.mHandler.sendEmptyMessage(1);
            }
        };
        this.publishThread = thread;
        thread.start();
    }

    private void stopSync() {
        Thread thread = this.publishThread;
        if (thread == null) {
            return;
        }
        thread.interrupt();
        this.publishThread = null;
    }

    private String textToHtml() {
        String obj = this.mEditContent.getText().toString();
        for (String str : this.replacePathMap.keySet()) {
            obj = obj.replace(str, this.replacePathMap.get(str));
        }
        return obj.replaceAll("<img>(.*?)</img>", "<img src=\"$1\" alt= />");
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void topicSend() {
        stopSync();
        String str = (String) SharedPreferencesUtil.get(this, Constans.Token);
        String valueOf = String.valueOf(System.currentTimeMillis());
        String obj = this.mEditTitle.getText().toString();
        String str2 = "<div>" + textToHtml() + "</div>";
        LogUtil.e("content:" + str2);
        this.baseRequest.getApi().topicSend(str, valueOf, this.mCategorieId, obj, str2).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<TopicSendModel>(MyApplication.getApplication(), null, false) { // from class: com.hismith.hismithapp.PublishActivity.3
            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onComplete() {
                super.onComplete();
                if (PublishActivity.this.mTxtComplete.getVisibility() == 8) {
                    PublishActivity.this.mProgress.setVisibility(8);
                    PublishActivity.this.mTxtComplete.setVisibility(0);
                }
                if (PublishActivity.this.mTxtPublish.getVisibility() == 8) {
                    PublishActivity.this.mTitleProgress.setVisibility(8);
                    PublishActivity.this.mTxtPublish.setVisibility(0);
                }
            }

            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onError(Throwable th) {
                super.onError(th);
                if (PublishActivity.this.mTxtComplete.getVisibility() == 8) {
                    PublishActivity.this.mProgress.setVisibility(8);
                    PublishActivity.this.mTxtComplete.setVisibility(0);
                }
                if (PublishActivity.this.mTxtPublish.getVisibility() == 8) {
                    PublishActivity.this.mTitleProgress.setVisibility(8);
                    PublishActivity.this.mTxtPublish.setVisibility(0);
                }
            }

            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(TopicSendModel topicSendModel) {
                if (topicSendModel.getSuccess() == 1) {
                    ToastUtil.showShortToastCenter(PublishActivity.this, LanguageUtil.getLanguageByKey(PublishActivity.this, "published"));
                    PublishActivity publishActivity = PublishActivity.this;
                    SoftKeyboardUtil.hideSoftKeyboard(publishActivity, (List<View>) publishActivity.mEditList);
                    PublishActivity.this.finish();
                }
            }
        });
    }

    private void upload(Map<String, RequestBody> map) {
        this.fileRequest.getApi().uploadFile(map).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<UploadModel>(this, null, false) { // from class: com.hismith.hismithapp.PublishActivity.2
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(UploadModel uploadModel) {
                if (uploadModel.getSuccess() == 1) {
                    String fileUrl = uploadModel.getFileUrl();
                    LogUtil.e("file:" + uploadModel.getRequestId() + ", fileUrl:" + fileUrl);
                    PublishActivity.this.replacePathMap.put(uploadModel.getRequestId(), fileUrl);
                }
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void uploadFile(Uri uri) {
        if (uri != null) {
            String str = (String) SharedPreferencesUtil.get(this, Constans.Token);
            HashMap hashMap = new HashMap();
            RequestBodyUtil.addMultiPart(this, hashMap, Constans.Token, str);
            RequestBodyUtil.addMultiPart(this, hashMap, NativeProtocol.WEB_DIALOG_ACTION, "upload");
            RequestBodyUtil.addMultiPart(this, hashMap, "fileType", "image");
            String fileNameFromUri = RequestBodyUtil.getFileNameFromUri(this, uri);
            RequestBodyUtil.addMultiPart(this, hashMap, "requestId", fileNameFromUri);
            RequestBodyUtil.addMultiPartUri(this, hashMap, fileNameFromUri, uri);
            upload(hashMap);
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void uploadFile(File file) {
        if (file == null || !file.exists()) {
            return;
        }
        String str = (String) SharedPreferencesUtil.get(this, Constans.Token);
        LogUtil.e("上传图片file：" + file.getPath());
        HashMap hashMap = new HashMap();
        RequestBodyUtil.addMultiPart(this, hashMap, Constans.Token, str);
        RequestBodyUtil.addMultiPart(this, hashMap, NativeProtocol.WEB_DIALOG_ACTION, "upload");
        RequestBodyUtil.addMultiPart(this, hashMap, "fileType", "image");
        RequestBodyUtil.addMultiPart(this, hashMap, "requestId", file.getName());
        RequestBodyUtil.addMultiPart(this, hashMap, "file", file);
        upload(hashMap);
    }

    @Override // android.text.TextWatcher
    public void afterTextChanged(Editable editable) {
        if (this.isImage) {
            this.isImage = false;
            if (this.mTagCount - StringUtils.countString(editable.toString(), PictureAndTextEditorView.mBitmapTag) == 1) {
                try {
                    if (editable.length() > this.startIndex) {
                        int length = editable.length();
                        int i = this.endIndex;
                        if (length > i - 1) {
                            editable.delete(this.startIndex, i);
                        }
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }
    }

    @Override // android.text.TextWatcher
    public void beforeTextChanged(CharSequence charSequence, int i, int i2, int i3) {
        int lastIndexOf;
        String obj = charSequence.toString();
        if (i2 != 1 || !obj.substring(i, i + 1).equals(PictureAndTextEditorView.mBitmapTag) || i <= 0 || (lastIndexOf = obj.lastIndexOf(PictureAndTextEditorView.mBitmapTag, i - 1)) <= 0) {
            this.isImage = false;
            return;
        }
        this.isImage = true;
        this.mTagCount = StringUtils.countString(obj, PictureAndTextEditorView.mBitmapTag);
        this.startIndex = lastIndexOf;
        this.endIndex = i;
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        int id = view.getId();
        if (id == R.id.title_img_back) {
            finish();
            return;
        }
        if (id != R.id.publish_txt_publish && id != R.id.publish_txt_complete) {
            if (id == R.id.publish_img_camera) {
                PictureSelectorManager.getInstance(this).openCamera(new PictureSelectorManager.OnCameraListener() { // from class: com.hismith.hismithapp.PublishActivity.5
                    @Override // com.hismith.hismithapp.picker.PictureSelectorManager.OnCameraListener
                    public void onCameraResult(List<LocalMedia> list) {
                        for (int i = 0; list != null && i < list.size(); i++) {
                            if (list.get(i).isCompressed()) {
                                String compressPath = list.get(i).getCompressPath();
                                LogUtil.e("新图片库拍摄压缩后：" + compressPath);
                                PublishActivity.this.mEditContent.insertBitmap(compressPath, true);
                                File file = new File(compressPath);
                                if (!PublishActivity.this.replacePathMap.containsKey(file.getName())) {
                                    PublishActivity.this.replacePathMap.put(file.getName(), "");
                                    PublishActivity.this.uploadFile(file);
                                }
                            } else {
                                String realPath = list.get(i).getRealPath();
                                PublishActivity.this.mEditContent.insertBitmap(realPath, true);
                                File file2 = new File(realPath);
                                if (!PublishActivity.this.replacePathMap.containsKey(file2.getName())) {
                                    PublishActivity.this.replacePathMap.put(file2.getName(), "");
                                    PublishActivity.this.uploadFile(file2);
                                }
                            }
                            PublishActivity publishActivity = PublishActivity.this;
                            SoftKeyboardUtil.openKeyBoard(publishActivity, publishActivity.mEditContent);
                        }
                    }
                });
                return;
            } else {
                if (id == R.id.publish_img_album) {
                    this.pictureResultManager.openGallery(new PictureResultManager.OnGalleryListener() { // from class: com.hismith.hismithapp.PublishActivity.6
                        @Override // com.hismith.hismithapp.manager.PictureResultManager.OnGalleryListener
                        public void onGalleryResult(Uri uri) {
                            if (uri != null) {
                                PublishActivity.this.mEditContent.insertUriBitmap(uri, true);
                                PublishActivity.this.replacePathMap.put(RequestBodyUtil.getFileNameFromUri(PublishActivity.this, uri), "");
                                PublishActivity.this.uploadFile(uri);
                            }
                            PublishActivity publishActivity = PublishActivity.this;
                            SoftKeyboardUtil.openKeyBoard(publishActivity, publishActivity.mEditContent);
                        }

                        @Override // com.hismith.hismithapp.manager.PictureResultManager.OnGalleryListener
                        public void onGalleryResultFile(String str) {
                            if (!TextUtils.isEmpty(str)) {
                                PublishActivity.this.mEditContent.insertBitmap(str, true);
                                File file = new File(str);
                                if (!PublishActivity.this.replacePathMap.containsKey(file.getName())) {
                                    PublishActivity.this.replacePathMap.put(file.getName(), "");
                                    PublishActivity.this.uploadFile(file);
                                }
                            }
                            PublishActivity publishActivity = PublishActivity.this;
                            SoftKeyboardUtil.openKeyBoard(publishActivity, publishActivity.mEditContent);
                        }
                    });
                    return;
                }
                return;
            }
        }
        String trim = this.mEditTitle.getText().toString().trim();
        String trim2 = this.mEditContent.getText().toString().trim();
        if (TextUtils.isEmpty(trim) || TextUtils.isEmpty(trim2)) {
            return;
        }
        this.mTxtComplete.setVisibility(8);
        this.mProgress.setVisibility(0);
        this.mTxtPublish.setVisibility(8);
        this.mTitleProgress.setVisibility(0);
        startSync();
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, androidx.core.app.ComponentActivity, android.app.Activity
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        setContentView(R.layout.activity_publish);
        initView();
        init();
        setListener();
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.appcompat.app.AppCompatActivity, androidx.fragment.app.FragmentActivity, android.app.Activity
    protected void onDestroy() {
        super.onDestroy();
        CompositeDisposable compositeDisposable = this.mCompositeDisposable;
        if (compositeDisposable != null) {
            compositeDisposable.clear();
            this.mCompositeDisposable = null;
        }
        stopSync();
    }

    @Override // android.view.View.OnFocusChangeListener
    public void onFocusChange(View view, boolean z) {
        LinearLayout linearLayout = this.mLayoutAction;
        if (z) {
            linearLayout.setVisibility(0);
        } else {
            linearLayout.setVisibility(8);
        }
    }

    @Override // android.text.TextWatcher
    public void onTextChanged(CharSequence charSequence, int i, int i2, int i3) {
    }
}
