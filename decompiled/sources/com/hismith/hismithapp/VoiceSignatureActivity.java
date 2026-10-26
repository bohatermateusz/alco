package com.hismith.hismithapp;

import android.content.Intent;
import android.media.MediaPlayer;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.text.TextUtils;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.activity.OnBackPressedCallback;
import com.bumptech.glide.load.Key;
import com.facebook.internal.NativeProtocol;
import com.facebook.share.internal.ShareConstants;
import com.google.gson.Gson;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.http.HttpModel;
import com.hismith.hismithapp.http.RetrofitRequest;
import com.hismith.hismithapp.http.RxSubscribe;
import com.hismith.hismithapp.model.UploadModel;
import com.hismith.hismithapp.model.UserModel;
import com.hismith.hismithapp.utils.AppUtil;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.RecorderUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import com.hismith.hismithapp.utils.ToastUtil;
import com.hismith.hismithapp.view.GradientRoundProgress;
import com.tbruyelle.rxpermissions2.Permission;
import com.tbruyelle.rxpermissions2.RxPermissions;
import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.disposables.Disposable;
import io.reactivex.functions.Consumer;
import io.reactivex.schedulers.Schedulers;
import java.io.File;
import java.net.URLEncoder;
import java.util.HashMap;
import java.util.Map;
import okhttp3.MediaType;
import okhttp3.RequestBody;

/* loaded from: classes3.dex */
public class VoiceSignatureActivity extends BaseActivity implements View.OnClickListener, MediaPlayer.OnPreparedListener, MediaPlayer.OnCompletionListener {
    private static final int RECORDER_CANCEL = 2;
    private static final int RECORDER_DELAY_MILLIS = 50;
    private static final int RECORDER_MAX_LENGTH = 30000;
    private static final int RECORDER_START = 0;
    private static final int RECORDER_STOP = 1;
    private RetrofitRequest baseRequest;
    private RetrofitRequest fileRequest;
    private String fileUrl;
    private Gson gson;
    private ImageView mImgPlay;
    private ImageView mImgRecord;
    private ImageView mImgRepeal;
    private ImageView mImgSave;
    private LinearLayout mLayoutDelete;
    private LinearLayout mLayoutSave;
    private GradientRoundProgress mProgressTime;
    private ImageView mTitleImgBack;
    private TextView mTitleTxtName;
    private TextView mTxtSecondes;
    private MediaPlayer mediaPlayer;
    private int voiceLength;
    private RecorderUtil recorderUtil = new RecorderUtil();
    private int recorderSeconds = -1;
    private boolean isRecorder = false;
    private boolean isPlay = false;
    private int action = -1;
    private Handler mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.VoiceSignatureActivity.1
        @Override // android.os.Handler.Callback
        public boolean handleMessage(Message message) {
            int i = message.what;
            if (i == 0) {
                VoiceSignatureActivity.access$012(VoiceSignatureActivity.this, 50);
                VoiceSignatureActivity.this.mProgressTime.setProgress(VoiceSignatureActivity.this.recorderSeconds);
                VoiceSignatureActivity.this.mTxtSecondes.setText((VoiceSignatureActivity.this.recorderSeconds / 1000) + "''");
                if (!VoiceSignatureActivity.this.isRecorder) {
                    if (VoiceSignatureActivity.this.mLayoutDelete.getVisibility() == 0) {
                        VoiceSignatureActivity.this.mLayoutDelete.setVisibility(4);
                    }
                    VoiceSignatureActivity.this.recorderUtil.startRecording();
                    VoiceSignatureActivity.this.isRecorder = true;
                }
                int i2 = VoiceSignatureActivity.this.recorderSeconds;
                VoiceSignatureActivity voiceSignatureActivity = VoiceSignatureActivity.this;
                if (i2 >= VoiceSignatureActivity.RECORDER_MAX_LENGTH) {
                    voiceSignatureActivity.mHandler.removeMessages(0);
                    VoiceSignatureActivity.this.mHandler.sendEmptyMessage(1);
                } else {
                    voiceSignatureActivity.mHandler.sendEmptyMessageDelayed(0, 50L);
                }
            } else if (i == 1) {
                VoiceSignatureActivity.this.recorderUtil.stopRecording();
                VoiceSignatureActivity.this.isRecorder = false;
                VoiceSignatureActivity.this.recorderSeconds = -1;
                VoiceSignatureActivity.this.mImgPlay.setVisibility(0);
                VoiceSignatureActivity.this.mImgRecord.setVisibility(8);
                VoiceSignatureActivity.this.mLayoutSave.setVisibility(0);
                if (MyApplication.getApplication().getmUserModel() != null && !TextUtils.isEmpty(MyApplication.getApplication().getmUserModel().getVoice())) {
                    VoiceSignatureActivity.this.mLayoutDelete.setVisibility(0);
                }
            } else if (i == 2) {
                VoiceSignatureActivity.this.isRecorder = false;
            }
            return false;
        }
    });
    private View.OnTouchListener OnVoiceStartTouchListener = new View.OnTouchListener() { // from class: com.hismith.hismithapp.VoiceSignatureActivity.7
        @Override // android.view.View.OnTouchListener
        public boolean onTouch(View view, MotionEvent motionEvent) {
            int action = motionEvent.getAction();
            if (action == 0) {
                VoiceSignatureActivity.this.action = 0;
                VoiceSignatureActivity.this.initRecordPermission();
            } else if (action == 1 || action == 3) {
                VoiceSignatureActivity.this.action = 1;
                if (VoiceSignatureActivity.this.isRecorder) {
                    VoiceSignatureActivity.this.mHandler.removeCallbacksAndMessages(null);
                    VoiceSignatureActivity.this.mHandler.sendEmptyMessage(1);
                }
            }
            return true;
        }
    };

    static /* synthetic */ int access$012(VoiceSignatureActivity voiceSignatureActivity, int i) {
        int i2 = voiceSignatureActivity.recorderSeconds + i;
        voiceSignatureActivity.recorderSeconds = i2;
        return i2;
    }

    private void addMultiPart(Map<String, RequestBody> map, String str, Object obj) {
        if (obj instanceof String) {
            map.put(str, RequestBody.create(MediaType.parse("text/plain;charset=UTF-8"), (String) obj));
            return;
        }
        if (obj instanceof File) {
            try {
                String encode = URLEncoder.encode(((File) obj).getName(), Key.STRING_CHARSET_NAME);
                Log.i("UploadFile", "getRequestParams: " + encode);
                map.put(str + "\"; filename= \"" + encode + "", RequestBody.create(MediaType.parse("application/octet-stream"), (File) obj));
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    private void deleteVoice() {
        this.baseRequest.getApi().setVoice((String) SharedPreferencesUtil.get(this, Constans.Token), "", 0).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<HttpModel>(this, this.progressDialog, true) { // from class: com.hismith.hismithapp.VoiceSignatureActivity.4
            @Override // com.hismith.hismithapp.http.RxSubscribe
            protected void onSuccess(HttpModel httpModel) {
                if (httpModel.getSuccess() == 1) {
                    ToastUtil.showShortToastCenter(VoiceSignatureActivity.this, LanguageUtil.getLanguageByKey(VoiceSignatureActivity.this, "deletedMsg"));
                    VoiceSignatureActivity.this.mImgPlay.setVisibility(8);
                    VoiceSignatureActivity.this.mLayoutDelete.setVisibility(4);
                    VoiceSignatureActivity.this.mProgressTime.setProgress(0.0f);
                    VoiceSignatureActivity.this.mTxtSecondes.setText("0''");
                    MyApplication.getApplication().getmUserModel().setVoice("");
                    MyApplication.getApplication().getmUserModel().setVoiceLength(0);
                    VoiceSignatureActivity.this.sendBroadcast(new Intent(Constans.RECEIVER_USER_INFO_CHANGE));
                }
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void getMyUserInfo() {
        this.baseRequest.getApi().getMyInfo((String) SharedPreferencesUtil.get(this, Constans.Token), MyApplication.FCM_TOKEN, AppUtil.getAppVersionName(this)).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<UserModel>(this, this.progressDialog, false) { // from class: com.hismith.hismithapp.VoiceSignatureActivity.3
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(UserModel userModel) {
                if (userModel.getSuccess() == 1) {
                    boolean isEmpty = TextUtils.isEmpty(userModel.getVoice());
                    VoiceSignatureActivity voiceSignatureActivity = VoiceSignatureActivity.this;
                    if (isEmpty) {
                        voiceSignatureActivity.mImgPlay.setVisibility(8);
                        VoiceSignatureActivity.this.mLayoutDelete.setVisibility(4);
                    } else {
                        voiceSignatureActivity.mImgPlay.setVisibility(0);
                        VoiceSignatureActivity.this.mLayoutDelete.setVisibility(0);
                        VoiceSignatureActivity.this.mProgressTime.setProgress(userModel.getVoiceLength() * 1000);
                        VoiceSignatureActivity.this.mTxtSecondes.setText(userModel.getVoiceLength() + "''");
                        VoiceSignatureActivity.this.mLayoutDelete.setVisibility(0);
                    }
                    MyApplication.getApplication().setmUserModel(userModel);
                    VoiceSignatureActivity.this.sendBroadcast(new Intent(Constans.RECEIVER_USER_INFO_CHANGE));
                }
            }
        });
    }

    private void init() {
        this.baseRequest = new RetrofitRequest();
        this.fileRequest = new RetrofitRequest(RetrofitRequest.EnumRequest.REQUEST_FILE);
        this.gson = new Gson();
        this.mProgressTime.setMax(30000.0f);
        UserModel userModel = MyApplication.getApplication().getmUserModel();
        if (userModel == null || TextUtils.isEmpty(userModel.getVoice())) {
            return;
        }
        this.mImgPlay.setVisibility(0);
        this.mLayoutDelete.setVisibility(0);
        this.mProgressTime.setProgress(userModel.getVoiceLength() * 1000);
        this.mTxtSecondes.setText(userModel.getVoiceLength() + "''");
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void initRecordPermission() {
        Disposable subscribe = new RxPermissions(this).requestEachCombined("android.permission.RECORD_AUDIO").subscribe(new Consumer() { // from class: com.hismith.hismithapp.VoiceSignatureActivity$$ExternalSyntheticLambda0
            @Override // io.reactivex.functions.Consumer
            public final void accept(Object obj) {
                VoiceSignatureActivity.this.m577xa257f561((Permission) obj);
            }
        });
        if (this.mCompositeDisposable == null || this.mCompositeDisposable.isDisposed()) {
            return;
        }
        this.mCompositeDisposable.add(subscribe);
    }

    private void initView() {
        this.mTitleImgBack = (ImageView) findViewById(R.id.title_img_back);
        this.mTitleTxtName = (TextView) findViewById(R.id.title_txt_name);
        this.mProgressTime = (GradientRoundProgress) findViewById(R.id.voice_progress_seconds);
        this.mTxtSecondes = (TextView) findViewById(R.id.voice_txt_secondes);
        this.mImgPlay = (ImageView) findViewById(R.id.voice_img_play);
        this.mImgRecord = (ImageView) findViewById(R.id.img_record);
        this.mImgRepeal = (ImageView) findViewById(R.id.img_repeal);
        this.mImgSave = (ImageView) findViewById(R.id.img_save);
        this.mLayoutDelete = (LinearLayout) findViewById(R.id.layout_delete);
        this.mLayoutSave = (LinearLayout) findViewById(R.id.layout_save);
        LanguageUtil.setText(this, this.mTitleTxtName, "voiceRecording");
        this.mTxtSecondes.setText("0''");
    }

    private void playVoice(String str) {
        if (TextUtils.isEmpty(str)) {
            return;
        }
        releasePlay();
        try {
            MediaPlayer mediaPlayer = new MediaPlayer();
            this.mediaPlayer = mediaPlayer;
            mediaPlayer.setDataSource(str);
            this.mediaPlayer.setOnPreparedListener(this);
            this.mediaPlayer.setOnCompletionListener(this);
            this.mediaPlayer.prepareAsync();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void releasePlay() {
        MediaPlayer mediaPlayer = this.mediaPlayer;
        if (mediaPlayer != null) {
            mediaPlayer.release();
            this.mediaPlayer = null;
        }
    }

    private void setListener() {
        this.mTitleImgBack.setOnClickListener(this);
        this.mImgRecord.setOnTouchListener(this.OnVoiceStartTouchListener);
        this.mImgPlay.setOnClickListener(this);
        this.mImgRepeal.setOnClickListener(this);
        this.mImgSave.setOnClickListener(this);
        this.mLayoutDelete.setOnClickListener(this);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void setVoice() {
        if (TextUtils.isEmpty(this.fileUrl)) {
            return;
        }
        this.baseRequest.getApi().setVoice((String) SharedPreferencesUtil.get(this, Constans.Token), this.fileUrl, Integer.valueOf(this.voiceLength)).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<HttpModel>(this, this.progressDialog, false) { // from class: com.hismith.hismithapp.VoiceSignatureActivity.5
            @Override // com.hismith.hismithapp.http.RxSubscribe
            protected void onSuccess(HttpModel httpModel) {
                if (httpModel.getSuccess() == 1) {
                    VoiceSignatureActivity.this.mImgRecord.setVisibility(0);
                    VoiceSignatureActivity.this.mLayoutSave.setVisibility(8);
                    VoiceSignatureActivity.this.getMyUserInfo();
                }
            }
        });
    }

    private void uploadFile() {
        if (TextUtils.isEmpty(this.recorderUtil.getFilePath())) {
            return;
        }
        try {
            final File file = new File(this.recorderUtil.getFilePath());
            if (file.exists() && !file.isDirectory()) {
                Object obj = (String) SharedPreferencesUtil.get(this, Constans.Token);
                Map<String, RequestBody> hashMap = new HashMap<>();
                addMultiPart(hashMap, Constans.Token, obj);
                addMultiPart(hashMap, NativeProtocol.WEB_DIALOG_ACTION, "upload");
                addMultiPart(hashMap, "fileType", ShareConstants.WEB_DIALOG_PARAM_MEDIA);
                addMultiPart(hashMap, "file", file);
                this.fileRequest.getApi().uploadFile(hashMap).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<UploadModel>(this, this.progressDialog) { // from class: com.hismith.hismithapp.VoiceSignatureActivity.6
                    /* JADX INFO: Access modifiers changed from: protected */
                    @Override // com.hismith.hismithapp.http.RxSubscribe
                    public void onSuccess(UploadModel uploadModel) {
                        if (uploadModel.getSuccess() == 1) {
                            VoiceSignatureActivity.this.fileUrl = uploadModel.getFileUrl();
                            VoiceSignatureActivity voiceSignatureActivity = VoiceSignatureActivity.this;
                            voiceSignatureActivity.voiceLength = (int) voiceSignatureActivity.recorderUtil.getTimeInterval();
                            LogUtil.e("上传文件：" + VoiceSignatureActivity.this.fileUrl + ", length:" + file.length());
                            VoiceSignatureActivity.this.setVoice();
                        }
                    }
                });
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /* renamed from: lambda$initRecordPermission$0$com-hismith-hismithapp-VoiceSignatureActivity, reason: not valid java name */
    /* synthetic */ void m577xa257f561(Permission permission) throws Exception {
        if (permission.granted) {
            if (this.action == 0) {
                this.mHandler.sendEmptyMessage(0);
            }
        } else {
            if (permission.shouldShowRequestPermissionRationale) {
                return;
            }
            ToastUtil.showShortToast(this, LanguageUtil.getLanguageByKey(this, "microPermission"));
        }
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        int id = view.getId();
        if (id == R.id.title_img_back) {
            finish();
            return;
        }
        if (id == R.id.voice_img_play) {
            try {
                if (this.isPlay) {
                    releasePlay();
                    this.isPlay = false;
                    this.mImgPlay.setImageResource(R.mipmap.ic_voice_play);
                    return;
                } else {
                    if (this.mLayoutSave.getVisibility() == 0) {
                        playVoice(this.recorderUtil.getFilePath());
                    } else {
                        playVoice(MyApplication.getApplication().getmUserModel().getVoice());
                    }
                    this.isPlay = true;
                    this.mImgPlay.setImageResource(R.mipmap.ic_voice_pause);
                    return;
                }
            } catch (Exception e) {
                e.printStackTrace();
                return;
            }
        }
        if (id == R.id.layout_delete) {
            deleteVoice();
            return;
        }
        if (id != R.id.img_repeal) {
            if (id == R.id.img_save) {
                releasePlay();
                uploadFile();
                return;
            }
            return;
        }
        releasePlay();
        this.mImgRecord.setVisibility(0);
        this.mLayoutSave.setVisibility(8);
        if (MyApplication.getApplication().getmUserModel() == null || TextUtils.isEmpty(MyApplication.getApplication().getmUserModel().getVoice())) {
            this.mImgPlay.setVisibility(8);
            this.mProgressTime.setProgress(0.0f);
            this.mTxtSecondes.setText("0''");
            this.mLayoutDelete.setVisibility(8);
            return;
        }
        this.mProgressTime.setProgress(MyApplication.getApplication().getmUserModel().getVoiceLength() * 1000);
        this.mTxtSecondes.setText(MyApplication.getApplication().getmUserModel().getVoiceLength() + "''");
        this.mLayoutDelete.setVisibility(0);
        this.mImgPlay.setVisibility(0);
    }

    @Override // android.media.MediaPlayer.OnCompletionListener
    public void onCompletion(MediaPlayer mediaPlayer) {
        this.isPlay = false;
        this.mImgPlay.setImageResource(R.mipmap.ic_voice_play);
        LogUtil.e("播放完成");
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, androidx.core.app.ComponentActivity, android.app.Activity
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        getOnBackPressedDispatcher().addCallback(this, new OnBackPressedCallback(true) { // from class: com.hismith.hismithapp.VoiceSignatureActivity.2
            @Override // androidx.activity.OnBackPressedCallback
            public void handleOnBackPressed() {
                int visibility = VoiceSignatureActivity.this.mLayoutSave.getVisibility();
                VoiceSignatureActivity voiceSignatureActivity = VoiceSignatureActivity.this;
                if (visibility != 0) {
                    voiceSignatureActivity.finish();
                } else {
                    voiceSignatureActivity.mImgRecord.setVisibility(0);
                    VoiceSignatureActivity.this.mLayoutSave.setVisibility(8);
                }
            }
        });
        setContentView(R.layout.activity_voice_signature);
        initView();
        init();
        setListener();
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.appcompat.app.AppCompatActivity, androidx.fragment.app.FragmentActivity, android.app.Activity
    protected void onDestroy() {
        super.onDestroy();
        this.mHandler.removeCallbacksAndMessages(null);
        this.recorderUtil.cancelRecording();
        releasePlay();
    }

    @Override // android.media.MediaPlayer.OnPreparedListener
    public void onPrepared(MediaPlayer mediaPlayer) {
        try {
            mediaPlayer.start();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
