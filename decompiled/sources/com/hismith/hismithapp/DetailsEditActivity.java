package com.hismith.hismithapp;

import android.content.BroadcastReceiver;
import android.content.ClipData;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.text.TextUtils;
import android.view.View;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.TextView;
import androidx.fragment.app.FragmentActivity;
import com.bumptech.glide.Glide;
import com.bumptech.glide.request.BaseRequestOptions;
import com.bumptech.glide.request.RequestOptions;
import com.facebook.internal.NativeProtocol;
import com.facebook.internal.ServerProtocol;
import com.google.gson.Gson;
import com.hismith.hismithapp.adapter.CommAdapter;
import com.hismith.hismithapp.adapter.XQuickViewHolder;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.dialog.ChoosePhotoDialog;
import com.hismith.hismithapp.dialog.DatePickerDialog;
import com.hismith.hismithapp.glide.GlideRoundTransform;
import com.hismith.hismithapp.http.HttpModel;
import com.hismith.hismithapp.http.RetrofitRequest;
import com.hismith.hismithapp.http.RxSubscribe;
import com.hismith.hismithapp.listener.SoftKeyBoardListener;
import com.hismith.hismithapp.manager.PictureResultManager;
import com.hismith.hismithapp.model.UploadModel;
import com.hismith.hismithapp.model.UserModel;
import com.hismith.hismithapp.picker.PictureSelectorManager;
import com.hismith.hismithapp.utils.AppUtil;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.FileUtil;
import com.hismith.hismithapp.utils.ImageUtil;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.RequestBodyUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import com.hismith.hismithapp.utils.ToastUtil;
import com.hismith.hismithapp.view.GridLayout;
import com.luck.picture.lib.config.PictureMimeType;
import com.luck.picture.lib.config.SelectMimeType;
import com.luck.picture.lib.entity.LocalMedia;
import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.schedulers.Schedulers;
import java.io.File;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import okhttp3.RequestBody;

/* loaded from: classes3.dex */
public class DetailsEditActivity extends BaseActivity implements View.OnClickListener, SoftKeyBoardListener.OnSoftKeyBoardChangeListener {
    private RetrofitRequest baseRequest;
    private Calendar calendar;
    private int clickPosition;
    private RetrofitRequest fileRequest;
    private Gson gson;
    private CommAdapter<String> mAdapter;
    private DatePickerDialog mDatePickerDialog;
    private TextView mEditEditBio;
    private TextView mEditEditBirthday;
    private TextView mEditEditEmail;
    private EditText mEditEditName;
    private EditText mEditEditPhone;
    private GridLayout mEditGridLayout;
    private LinearLayout mEditLayoutBio;
    private LinearLayout mEditLayoutHasVoiceSignature;
    private FrameLayout mEditLayoutVoiceSignature;
    private TextView mEditTxtBio;
    private TextView mEditTxtBirthday;
    private TextView mEditTxtEmail;
    private TextView mEditTxtFrom;
    private TextView mEditTxtFromLabel;
    private TextView mEditTxtGender;
    private TextView mEditTxtInterested;
    private TextView mEditTxtName;
    private TextView mEditTxtPhone;
    private TextView mEditTxtSave;
    private TextView mEditTxtVoiceLength;
    private TextView mEditTxtVoiceSignature;
    private TextView mEditTxtVoiceSignatureLabel;
    private String[] mGenderList;
    private String[] mImgArr;
    private String[] mInterestedInList;
    private LinearLayout mLayoutRoot;
    private ImageView mTitleImgBack;
    private TextView mTitleTxtName;
    private TextView mTxtGenderFemale;
    private TextView mTxtGenderMale;
    private TextView mTxtGenderTrans;
    private TextView mTxtInterestedFemale;
    private TextView mTxtInterestedMale;
    private TextView mTxtInterestedTrans;
    private Calendar maxCalendar;
    private PictureResultManager pictureResultManager;
    private SoftKeyBoardListener softKeyBoardListener;
    private Uri uritempFile;
    private int maxImg = 6;
    private int countryId = 0;
    private int cityId = 0;
    private int genderId = 0;
    private int interestedInId = -1;
    private String countryName = "";
    private String cityName = "";
    private final int REQUEST_COUNTRY = 111;
    private final int REQUEST_EMAIL = 222;
    private final int REQUEST_BIO = 333;
    private final int REQUEST_CAMERA = 777;
    private final int REQUEST_ALBUM = 888;
    private final int REQUEST_PHOTO_CLIP = 999;
    private BroadcastReceiver mReceiver = new BroadcastReceiver() { // from class: com.hismith.hismithapp.DetailsEditActivity.9
        @Override // android.content.BroadcastReceiver
        public void onReceive(Context context, Intent intent) {
            if (intent.getAction().equals(Constans.RECEIVER_USER_INFO_CHANGE)) {
                UserModel userModel = MyApplication.getApplication().getmUserModel();
                if (userModel == null || TextUtils.isEmpty(userModel.getVoice())) {
                    DetailsEditActivity.this.mEditTxtVoiceSignature.setVisibility(0);
                    DetailsEditActivity.this.mEditLayoutHasVoiceSignature.setVisibility(8);
                    DetailsEditActivity.this.mEditTxtVoiceLength.setText("");
                } else {
                    DetailsEditActivity.this.mEditTxtVoiceSignature.setVisibility(8);
                    DetailsEditActivity.this.mEditLayoutHasVoiceSignature.setVisibility(0);
                    DetailsEditActivity.this.mEditTxtVoiceLength.setText(userModel.getVoiceLength() + "s''");
                }
            }
        }
    };

    /* renamed from: com.hismith.hismithapp.DetailsEditActivity$1, reason: invalid class name */
    class AnonymousClass1 extends CommAdapter<String> {

        /* renamed from: com.hismith.hismithapp.DetailsEditActivity$1$1, reason: invalid class name and collision with other inner class name */
        class ViewOnClickListenerC00321 implements View.OnClickListener {
            final /* synthetic */ String val$item;
            final /* synthetic */ int val$position;

            ViewOnClickListenerC00321(int i, String str) {
                this.val$position = i;
                this.val$item = str;
            }

            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                DetailsEditActivity.this.clickPosition = this.val$position;
                ChoosePhotoDialog choosePhotoDialog = new ChoosePhotoDialog(DetailsEditActivity.this) { // from class: com.hismith.hismithapp.DetailsEditActivity.1.1.1
                    @Override // com.hismith.hismithapp.dialog.ChoosePhotoDialog
                    public void btnClickDelete() {
                        DetailsEditActivity.this.mImgArr[DetailsEditActivity.this.clickPosition] = "";
                        int i = DetailsEditActivity.this.clickPosition;
                        while (true) {
                            int length = DetailsEditActivity.this.mImgArr.length;
                            ViewOnClickListenerC00321 viewOnClickListenerC00321 = ViewOnClickListenerC00321.this;
                            if (i >= length) {
                                DetailsEditActivity.this.mAdapter.remove(DetailsEditActivity.this.clickPosition);
                                DetailsEditActivity.this.setUserPhotos();
                                return;
                            }
                            int length2 = DetailsEditActivity.this.mImgArr.length - 1;
                            ViewOnClickListenerC00321 viewOnClickListenerC003212 = ViewOnClickListenerC00321.this;
                            if (i < length2) {
                                DetailsEditActivity.this.mImgArr[i] = DetailsEditActivity.this.mImgArr[i + 1];
                            } else {
                                DetailsEditActivity.this.mImgArr[i] = "";
                            }
                            i++;
                        }
                    }

                    @Override // com.hismith.hismithapp.dialog.ChoosePhotoDialog
                    public void btnClickPhoto() {
                        DetailsEditActivity.this.pictureResultManager.openGallery(new PictureResultManager.OnGalleryListener() { // from class: com.hismith.hismithapp.DetailsEditActivity.1.1.1.2
                            @Override // com.hismith.hismithapp.manager.PictureResultManager.OnGalleryListener
                            public void onGalleryResult(Uri uri) {
                                if (uri == null) {
                                    LogUtil.e("onGalleryResult uri:null");
                                    return;
                                }
                                LogUtil.e("onGalleryResult uri:" + uri.toString());
                                DetailsEditActivity.this.mAdapter.set(DetailsEditActivity.this.clickPosition, (int) uri.toString());
                                if (DetailsEditActivity.this.mAdapter.getCount() < DetailsEditActivity.this.maxImg + 1 && DetailsEditActivity.this.clickPosition == DetailsEditActivity.this.mAdapter.getCount() - 1) {
                                    DetailsEditActivity.this.mAdapter.add("");
                                }
                                DetailsEditActivity.this.uploadFile(uri);
                            }

                            @Override // com.hismith.hismithapp.manager.PictureResultManager.OnGalleryListener
                            public void onGalleryResultFile(String str) {
                                if (TextUtils.isEmpty(str)) {
                                    LogUtil.e("onGalleryResult path:null");
                                    return;
                                }
                                LogUtil.e("onGalleryResult path:" + str);
                                DetailsEditActivity.this.mAdapter.set(DetailsEditActivity.this.clickPosition, (int) str);
                                if (DetailsEditActivity.this.mAdapter.getCount() < DetailsEditActivity.this.maxImg + 1 && DetailsEditActivity.this.clickPosition == DetailsEditActivity.this.mAdapter.getCount() - 1) {
                                    DetailsEditActivity.this.mAdapter.add("");
                                }
                                DetailsEditActivity.this.uploadFile(new File(str));
                            }
                        });
                    }

                    @Override // com.hismith.hismithapp.dialog.ChoosePhotoDialog
                    public void btnClickShoot() {
                        PictureSelectorManager.getInstance(DetailsEditActivity.this).openCamera(new PictureSelectorManager.OnCameraListener() { // from class: com.hismith.hismithapp.DetailsEditActivity.1.1.1.1
                            @Override // com.hismith.hismithapp.picker.PictureSelectorManager.OnCameraListener
                            public void onCameraResult(List<LocalMedia> list) {
                                for (int i = 0; i < list.size(); i++) {
                                    if (list.get(i).isCompressed()) {
                                        String compressPath = list.get(i).getCompressPath();
                                        LogUtil.e("新图片库拍摄压缩后：" + compressPath);
                                        DetailsEditActivity.this.mAdapter.set(DetailsEditActivity.this.clickPosition + i, (int) compressPath);
                                        if (DetailsEditActivity.this.mAdapter.getCount() < DetailsEditActivity.this.maxImg + 1 && DetailsEditActivity.this.clickPosition == DetailsEditActivity.this.mAdapter.getCount() - 1) {
                                            DetailsEditActivity.this.mAdapter.add("");
                                        }
                                        DetailsEditActivity.this.uploadFile(new File(compressPath));
                                    } else {
                                        String realPath = list.get(i).getRealPath();
                                        DetailsEditActivity.this.mAdapter.set(DetailsEditActivity.this.clickPosition + i, (int) realPath);
                                        if (DetailsEditActivity.this.mAdapter.getCount() < DetailsEditActivity.this.maxImg + 1 && DetailsEditActivity.this.clickPosition == DetailsEditActivity.this.mAdapter.getCount() - 1) {
                                            DetailsEditActivity.this.mAdapter.add("");
                                        }
                                        DetailsEditActivity.this.uploadFile(new File(realPath));
                                    }
                                }
                            }
                        });
                    }
                };
                if (TextUtils.isEmpty(this.val$item)) {
                    choosePhotoDialog.setHideDelete(true);
                } else {
                    choosePhotoDialog.setHideDelete(false);
                }
                choosePhotoDialog.show();
            }
        }

        AnonymousClass1(Context context, List list, int i) {
            super(context, list, i);
        }

        @Override // com.hismith.hismithapp.adapter.CommAdapter
        public void convert(XQuickViewHolder xQuickViewHolder, String str, int i) {
            LogUtil.e("CommAdapter item:" + str + ", position:" + i);
            if (!TextUtils.isEmpty(str)) {
                ImageView imageView = (ImageView) xQuickViewHolder.getView(R.id.item_topic_grid_image);
                ImageView imageView2 = (ImageView) xQuickViewHolder.getView(R.id.item_topic_grid_normal);
                ProgressBar progressBar = (ProgressBar) xQuickViewHolder.getView(R.id.item_topic_grid_progress);
                imageView2.setVisibility(8);
                progressBar.setVisibility(0);
                Glide.with((FragmentActivity) DetailsEditActivity.this).load(PictureMimeType.isContent(str) ? Uri.parse(str) : str).apply((BaseRequestOptions<?>) new RequestOptions().transform(new GlideRoundTransform(DetailsEditActivity.this, 8))).into(imageView);
            }
            xQuickViewHolder.setOnClickListener(new ViewOnClickListenerC00321(i, str));
        }
    }

    private void compressImage(String str, String str2) {
        Bitmap compressImageFromFile = ImageUtil.compressImageFromFile(str2, 1024.0f);
        compressImageFromFile.getByteCount();
        File compressImageFile = ImageUtil.compressImageFile(str, compressImageFromFile);
        LogUtil.e("文件压缩后大小：" + FileUtil.getFileSize(compressImageFile));
        uploadFile(compressImageFile);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void getMyUserInfo() {
        this.baseRequest.getApi().getMyInfo((String) SharedPreferencesUtil.get(this, Constans.Token), MyApplication.FCM_TOKEN, AppUtil.getAppVersionName(this)).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<UserModel>(this, this.progressDialog, false) { // from class: com.hismith.hismithapp.DetailsEditActivity.7
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(UserModel userModel) {
                if (userModel.getSuccess() == 1) {
                    MyApplication.getApplication().setmUserModel(userModel);
                    DetailsEditActivity.this.sendBroadcast(new Intent(Constans.RECEIVER_USER_INFO_CHANGE));
                }
            }
        });
    }

    private void init() {
        this.baseRequest = new RetrofitRequest();
        this.fileRequest = new RetrofitRequest(RetrofitRequest.EnumRequest.REQUEST_FILE);
        this.gson = new Gson();
        this.pictureResultManager = new PictureResultManager(this);
        this.softKeyBoardListener = new SoftKeyBoardListener(this);
        Calendar calendar = Calendar.getInstance();
        this.maxCalendar = calendar;
        calendar.add(1, -18);
        Calendar calendar2 = Calendar.getInstance();
        this.calendar = calendar2;
        calendar2.add(1, -18);
        this.mImgArr = new String[this.maxImg];
        int i = 0;
        this.mGenderList = new String[]{LanguageUtil.getLanguageByKey(this, "female"), LanguageUtil.getLanguageByKey(this, "male"), LanguageUtil.getLanguageByKey(this, "Trans")};
        this.mInterestedInList = new String[]{LanguageUtil.getLanguageByKey(this, "Girls"), LanguageUtil.getLanguageByKey(this, "Guys"), LanguageUtil.getLanguageByKey(this, "Trans")};
        UserModel userModel = MyApplication.getApplication().getmUserModel();
        if (userModel != null) {
            this.mEditEditEmail.setText(userModel.getEmail());
            this.mEditEditName.setText(userModel.getName());
            this.mEditTxtVoiceLength.setText(userModel.getVoiceLength() + "s''");
            if (!TextUtils.isEmpty(userModel.getCountry())) {
                this.mEditTxtFrom.setTextColor(getResources().getColor(R.color.white));
                boolean isEmpty = TextUtils.isEmpty(userModel.getCity());
                TextView textView = this.mEditTxtFrom;
                if (isEmpty) {
                    textView.setText(userModel.getCountry());
                } else {
                    textView.setText(userModel.getCountry() + ", " + userModel.getCity());
                }
            }
            if (!TextUtils.isEmpty(userModel.getPhotos())) {
                String[] split = userModel.getPhotos().split(",");
                for (int i2 = 0; i2 < split.length && i2 < this.maxImg; i2++) {
                    this.mImgArr[i2] = split[i2];
                }
            }
            if (!TextUtils.isEmpty(userModel.getBirthday())) {
                try {
                    this.calendar.setTime(new SimpleDateFormat("yyyy-MM-dd").parse(userModel.getBirthday()));
                } catch (ParseException e) {
                    e.printStackTrace();
                }
            }
            this.mEditEditBirthday.setText(userModel.getBirthday());
            this.genderId = userModel.getGender();
            this.interestedInId = userModel.getInterestedIn();
            this.cityId = userModel.getCityId();
            this.countryId = userModel.getCountryId();
            this.countryName = userModel.getCountry();
            this.cityName = userModel.getCity();
            selectGender(this.genderId);
            selectInterestedIn(this.interestedInId);
            this.mEditEditPhone.setText(userModel.getPhone());
            boolean isEmpty2 = TextUtils.isEmpty(userModel.getVoice());
            TextView textView2 = this.mEditTxtVoiceSignature;
            if (isEmpty2) {
                textView2.setVisibility(0);
                this.mEditLayoutHasVoiceSignature.setVisibility(8);
            } else {
                textView2.setVisibility(8);
                this.mEditLayoutHasVoiceSignature.setVisibility(0);
            }
            this.mEditEditBio.setText(userModel.getBios());
        }
        ArrayList arrayList = new ArrayList();
        while (true) {
            String[] strArr = this.mImgArr;
            if (i >= strArr.length) {
                break;
            }
            if (!TextUtils.isEmpty(strArr[i])) {
                arrayList.add(this.mImgArr[i]);
            }
            i++;
        }
        if (arrayList.size() < this.maxImg + 1) {
            arrayList.add("");
        }
        AnonymousClass1 anonymousClass1 = new AnonymousClass1(this, arrayList, R.layout.item_edit_image);
        this.mAdapter = anonymousClass1;
        this.mEditGridLayout.setAdapter(anonymousClass1);
        registerReceiver();
    }

    private void initView() {
        this.mLayoutRoot = (LinearLayout) findViewById(R.id.layout_root);
        this.mTitleImgBack = (ImageView) findViewById(R.id.title_img_back);
        this.mTitleTxtName = (TextView) findViewById(R.id.title_txt_name);
        this.mEditGridLayout = (GridLayout) findViewById(R.id.edit_grid_layout);
        this.mEditTxtName = (TextView) findViewById(R.id.edit_txt_name);
        this.mEditTxtVoiceSignatureLabel = (TextView) findViewById(R.id.edit_txt_voiceSignature_label);
        this.mEditTxtFromLabel = (TextView) findViewById(R.id.edit_txt_from_label);
        this.mEditTxtBirthday = (TextView) findViewById(R.id.edit_txt_birthday);
        this.mEditTxtGender = (TextView) findViewById(R.id.edit_txt_gender);
        this.mEditTxtPhone = (TextView) findViewById(R.id.edit_txt_phone);
        this.mEditTxtEmail = (TextView) findViewById(R.id.edit_txt_email);
        this.mEditEditName = (EditText) findViewById(R.id.edit_edit_name);
        this.mEditTxtVoiceSignature = (TextView) findViewById(R.id.edit_txt_voiceSignature);
        this.mEditTxtVoiceLength = (TextView) findViewById(R.id.edit_txt_voice_length);
        this.mEditLayoutHasVoiceSignature = (LinearLayout) findViewById(R.id.edit_layout_has_voiceSignature);
        this.mEditLayoutVoiceSignature = (FrameLayout) findViewById(R.id.edit_layout_voiceSignature);
        this.mEditTxtFrom = (TextView) findViewById(R.id.edit_txt_from);
        this.mEditEditBirthday = (TextView) findViewById(R.id.edit_edit_birthday);
        this.mEditEditPhone = (EditText) findViewById(R.id.edit_edit_phone);
        this.mEditEditEmail = (TextView) findViewById(R.id.edit_edit_email);
        this.mEditTxtInterested = (TextView) findViewById(R.id.edit_txt_interested);
        this.mTxtGenderFemale = (TextView) findViewById(R.id.txt_gender_female);
        this.mTxtGenderMale = (TextView) findViewById(R.id.txt_gender_male);
        this.mTxtGenderTrans = (TextView) findViewById(R.id.txt_gender_trans);
        this.mTxtInterestedFemale = (TextView) findViewById(R.id.txt_interested_female);
        this.mTxtInterestedMale = (TextView) findViewById(R.id.txt_interested_male);
        this.mTxtInterestedTrans = (TextView) findViewById(R.id.txt_interested_trans);
        this.mEditTxtSave = (TextView) findViewById(R.id.edit_txt_save);
        this.mEditLayoutBio = (LinearLayout) findViewById(R.id.edit_layout_bio);
        this.mEditTxtBio = (TextView) findViewById(R.id.edit_txt_bio);
        this.mEditEditBio = (TextView) findViewById(R.id.edit_edit_bio);
        LanguageUtil.setText(this, this.mTitleTxtName, "editPersonal");
        LanguageUtil.setText(this, this.mEditTxtName, "nickname");
        LanguageUtil.setText(this, this.mEditTxtVoiceSignatureLabel, "voiceSignature");
        LanguageUtil.setText(this, this.mEditTxtFromLabel, "from");
        this.mEditTxtFrom.setTextColor(getResources().getColor(R.color.white30));
        LanguageUtil.setText(this, this.mEditTxtFrom, "setting");
        LanguageUtil.setText(this, this.mEditTxtBirthday, "birthday");
        LanguageUtil.setText(this, this.mEditTxtGender, "gender");
        LanguageUtil.setText(this, this.mEditTxtPhone, "phone");
        LanguageUtil.setText(this, this.mEditTxtEmail, "email");
        LanguageUtil.setText(this, this.mEditTxtVoiceSignature, "setting");
        LanguageUtil.setHint(this, this.mEditEditPhone, "pleaseEnterPhone");
        LanguageUtil.setText(this, this.mEditTxtInterested, "InterestedIn");
        LanguageUtil.setText(this, this.mEditTxtSave, "save");
        LanguageUtil.setText(this, this.mEditTxtBio, "bio");
        LanguageUtil.setText(this, this.mTxtGenderFemale, "female");
        LanguageUtil.setText(this, this.mTxtGenderMale, "male");
        LanguageUtil.setText(this, this.mTxtGenderTrans, "Trans");
        LanguageUtil.setText(this, this.mTxtInterestedFemale, "Girls");
        LanguageUtil.setText(this, this.mTxtInterestedMale, "Guys");
        LanguageUtil.setText(this, this.mTxtInterestedTrans, "Trans");
    }

    private void registerReceiver() {
        IntentFilter intentFilter = new IntentFilter();
        intentFilter.addAction(Constans.RECEIVER_USER_INFO_CHANGE);
        AppUtil.registerReceiver(this, this.mReceiver, intentFilter);
    }

    private void selectGender(int i) {
        if (i == 0) {
            this.genderId = i;
            this.mTxtGenderFemale.setBackgroundResource(R.drawable.bg_edit_style_18);
            this.mTxtGenderMale.setBackground(null);
            this.mTxtGenderTrans.setBackground(null);
            return;
        }
        if (i == 1) {
            this.genderId = i;
            this.mTxtGenderFemale.setBackground(null);
            this.mTxtGenderMale.setBackgroundResource(R.drawable.bg_edit_style_18);
            this.mTxtGenderTrans.setBackground(null);
            return;
        }
        if (i != 2) {
            return;
        }
        this.genderId = i;
        this.mTxtGenderFemale.setBackground(null);
        this.mTxtGenderMale.setBackground(null);
        this.mTxtGenderTrans.setBackgroundResource(R.drawable.bg_edit_style_18);
    }

    private void selectInterestedIn(int i) {
        if (i == 0) {
            this.interestedInId = i;
            this.mTxtInterestedFemale.setBackgroundResource(R.drawable.bg_edit_style_18);
            this.mTxtInterestedMale.setBackground(null);
            this.mTxtInterestedTrans.setBackground(null);
            return;
        }
        if (i == 1) {
            this.interestedInId = i;
            this.mTxtInterestedFemale.setBackground(null);
            this.mTxtInterestedMale.setBackgroundResource(R.drawable.bg_edit_style_18);
            this.mTxtInterestedTrans.setBackground(null);
            return;
        }
        if (i != 2) {
            return;
        }
        this.interestedInId = i;
        this.mTxtInterestedFemale.setBackground(null);
        this.mTxtInterestedMale.setBackground(null);
        this.mTxtInterestedTrans.setBackgroundResource(R.drawable.bg_edit_style_18);
    }

    private void setListener() {
        this.mTitleImgBack.setOnClickListener(this);
        this.mEditLayoutVoiceSignature.setOnClickListener(this);
        this.mEditTxtFrom.setOnClickListener(this);
        this.mEditEditBirthday.setOnClickListener(this);
        this.mEditEditEmail.setOnClickListener(this);
        this.mEditTxtSave.setOnClickListener(this);
        this.mEditLayoutBio.setOnClickListener(this);
        this.mTxtGenderFemale.setOnClickListener(this);
        this.mTxtGenderMale.setOnClickListener(this);
        this.mTxtGenderTrans.setOnClickListener(this);
        this.mTxtInterestedFemale.setOnClickListener(this);
        this.mTxtInterestedMale.setOnClickListener(this);
        this.mTxtInterestedTrans.setOnClickListener(this);
        this.softKeyBoardListener.setListener(this);
        this.mEditEditName.setOnFocusChangeListener(new View.OnFocusChangeListener() { // from class: com.hismith.hismithapp.DetailsEditActivity.2
            @Override // android.view.View.OnFocusChangeListener
            public void onFocusChange(View view, boolean z) {
                DetailsEditActivity detailsEditActivity = DetailsEditActivity.this;
                if (z) {
                    detailsEditActivity.mEditEditName.setGravity(8388627);
                } else {
                    detailsEditActivity.mEditEditName.setGravity(8388629);
                }
            }
        });
        this.mEditEditPhone.setOnFocusChangeListener(new View.OnFocusChangeListener() { // from class: com.hismith.hismithapp.DetailsEditActivity.3
            @Override // android.view.View.OnFocusChangeListener
            public void onFocusChange(View view, boolean z) {
                DetailsEditActivity detailsEditActivity = DetailsEditActivity.this;
                if (z) {
                    detailsEditActivity.mEditEditPhone.setGravity(8388627);
                } else {
                    detailsEditActivity.mEditEditPhone.setGravity(8388629);
                }
            }
        });
    }

    private void setUserInfo() {
        String str = (String) SharedPreferencesUtil.get(this, Constans.Token);
        final String trim = this.mEditEditName.getText().toString().trim();
        final String obj = this.mEditEditBirthday.getText().toString();
        final String trim2 = this.mEditEditPhone.getText().toString().trim();
        this.baseRequest.getApi().setUserInfo(str, trim, Integer.valueOf(this.countryId), Integer.valueOf(this.cityId), obj, Integer.valueOf(this.genderId), Integer.valueOf(this.interestedInId), trim2).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<HttpModel>(this, this.progressDialog) { // from class: com.hismith.hismithapp.DetailsEditActivity.5
            @Override // com.hismith.hismithapp.http.RxSubscribe
            protected void onSuccess(HttpModel httpModel) {
                if (httpModel.getSuccess() == 1) {
                    ToastUtil.showShortToastCenter(DetailsEditActivity.this, LanguageUtil.getLanguageByKey(DetailsEditActivity.this, "saveed"));
                    UserModel userModel = MyApplication.getApplication().getmUserModel();
                    if (userModel != null) {
                        userModel.setName(trim);
                        userModel.setGender(DetailsEditActivity.this.genderId);
                        userModel.setInterestedIn(DetailsEditActivity.this.interestedInId);
                        userModel.setBirthday(obj);
                        userModel.setPhone(trim2);
                        userModel.setCountry(DetailsEditActivity.this.countryName);
                        userModel.setCountryId(DetailsEditActivity.this.countryId);
                        userModel.setCity(DetailsEditActivity.this.cityName);
                        userModel.setCityId(DetailsEditActivity.this.cityId);
                        MyApplication.getApplication().setmUserModel(userModel);
                        DetailsEditActivity.this.sendBroadcast(new Intent(Constans.RECEIVER_USER_INFO_CHANGE));
                    }
                }
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void setUserPhotos() {
        String str = (String) SharedPreferencesUtil.get(this, Constans.Token);
        String str2 = "";
        int i = 0;
        while (true) {
            String[] strArr = this.mImgArr;
            if (i >= strArr.length) {
                this.baseRequest.getApi().setUserInfo(str, str2).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<HttpModel>(this, this.progressDialog, true) { // from class: com.hismith.hismithapp.DetailsEditActivity.6
                    @Override // com.hismith.hismithapp.http.RxSubscribe
                    protected void onSuccess(HttpModel httpModel) {
                        if (httpModel.getSuccess() == 1) {
                            DetailsEditActivity.this.getMyUserInfo();
                        }
                    }
                });
                return;
            }
            if (!TextUtils.isEmpty(strArr[i])) {
                if (!TextUtils.isEmpty(str2)) {
                    str2 = str2 + ",";
                }
                str2 = str2 + this.mImgArr[i];
            }
            i++;
        }
    }

    private void showDatePickerDialog() {
        DatePickerDialog datePickerDialog = new DatePickerDialog(this, this.calendar.get(1), this.calendar.get(2), this.calendar.get(5)) { // from class: com.hismith.hismithapp.DetailsEditActivity.4
            @Override // com.hismith.hismithapp.dialog.DatePickerDialog
            public void btnSubmit(int i, int i2, int i3) {
                DetailsEditActivity.this.mEditEditBirthday.setText(i + "-" + (i2 < 10 ? new StringBuilder("0") : new StringBuilder("")).append(i2).toString() + "-" + (i3 < 10 ? new StringBuilder("0").append(i3) : new StringBuilder("").append(i3)).toString());
                DetailsEditActivity.this.calendar.set(i, i2 - 1, i3);
            }
        };
        this.mDatePickerDialog = datePickerDialog;
        datePickerDialog.showDialog();
    }

    private void unRegisterReceiver() {
        try {
            unregisterReceiver(this.mReceiver);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void upload(Map<String, RequestBody> map) {
        this.fileRequest.getApi().uploadFile(map).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<UploadModel>(this, this.progressDialog, true) { // from class: com.hismith.hismithapp.DetailsEditActivity.8
            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onError(Throwable th) {
                super.onError(th);
            }

            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(UploadModel uploadModel) {
                if (uploadModel.getSuccess() != 1) {
                    return;
                }
                String fileUrl = uploadModel.getFileUrl();
                int i = 0;
                while (true) {
                    int count = DetailsEditActivity.this.mAdapter.getCount();
                    DetailsEditActivity detailsEditActivity = DetailsEditActivity.this;
                    if (i >= count) {
                        detailsEditActivity.setUserPhotos();
                        return;
                    }
                    String str = (String) detailsEditActivity.mAdapter.getData().get(i);
                    if (str != null) {
                        if (PictureMimeType.isContent(str)) {
                            if (str.equals(uploadModel.getRequestId())) {
                                DetailsEditActivity.this.mImgArr[i] = fileUrl;
                            }
                        } else if (new File((String) DetailsEditActivity.this.mAdapter.getData().get(i)).getName().equals(uploadModel.getRequestId())) {
                            DetailsEditActivity.this.mImgArr[i] = fileUrl;
                        }
                    }
                    i++;
                }
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void uploadFile(Uri uri) {
        String str = (String) SharedPreferencesUtil.get(this, Constans.Token);
        if (uri != null) {
            LogUtil.e("上传图片file：" + uri);
            String fileNameFromUri = RequestBodyUtil.getFileNameFromUri(this, uri);
            HashMap hashMap = new HashMap();
            RequestBodyUtil.addMultiPart(this, hashMap, Constans.Token, str);
            RequestBodyUtil.addMultiPart(this, hashMap, NativeProtocol.WEB_DIALOG_ACTION, "upload");
            RequestBodyUtil.addMultiPart(this, hashMap, "requestId", uri.toString());
            RequestBodyUtil.addMultiPart(this, hashMap, "fileType", "image");
            RequestBodyUtil.addMultiPartUri(this, hashMap, fileNameFromUri, uri);
            upload(hashMap);
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void uploadFile(File file) {
        String str = (String) SharedPreferencesUtil.get(this, Constans.Token);
        if (file == null || !file.exists()) {
            return;
        }
        LogUtil.e("上传图片file：" + file.getPath() + ", 文件大小：" + file.length());
        HashMap hashMap = new HashMap();
        RequestBodyUtil.addMultiPart(this, hashMap, Constans.Token, str);
        RequestBodyUtil.addMultiPart(this, hashMap, NativeProtocol.WEB_DIALOG_ACTION, "upload");
        RequestBodyUtil.addMultiPart(this, hashMap, "requestId", file.getName());
        RequestBodyUtil.addMultiPart(this, hashMap, "fileType", "image");
        RequestBodyUtil.addMultiPart(this, hashMap, "file", file);
        upload(hashMap);
    }

    @Override // com.hismith.hismithapp.listener.SoftKeyBoardListener.OnSoftKeyBoardChangeListener
    public void keyBoardHide(int i) {
        if (this.mEditEditName.hasFocus()) {
            this.mEditEditName.clearFocus();
        }
        if (this.mEditEditPhone.hasFocus()) {
            this.mEditEditPhone.clearFocus();
        }
    }

    @Override // com.hismith.hismithapp.listener.SoftKeyBoardListener.OnSoftKeyBoardChangeListener
    public void keyBoardShow(int i) {
    }

    @Override // androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, android.app.Activity
    protected void onActivityResult(int i, int i2, Intent intent) {
        super.onActivityResult(i, i2, intent);
        if (i != 111) {
            if (i != 222) {
                if (i == 333 && MyApplication.getApplication().getmUserModel() != null) {
                    this.mEditEditBio.setText(MyApplication.getApplication().getmUserModel().getBios());
                    return;
                }
                return;
            }
            String str = (String) SharedPreferencesUtil.get(this, "email");
            if (MyApplication.getApplication() != null && MyApplication.getApplication().getmUserModel() != null) {
                str = MyApplication.getApplication().getmUserModel().getEmail();
            }
            this.mEditEditEmail.setText(str);
            return;
        }
        if (intent != null) {
            this.countryId = intent.getIntExtra(Constans.resultCountryId, 0);
            this.cityId = intent.getIntExtra(Constans.resultCityId, 0);
            this.countryName = intent.getStringExtra(Constans.resultCountryName);
            this.cityName = intent.getStringExtra(Constans.resultCityName);
            String str2 = this.countryName;
            if (str2 == null || str2.length() <= 0) {
                return;
            }
            this.mEditTxtFrom.setTextColor(getResources().getColor(R.color.white));
            String str3 = this.cityName;
            this.mEditTxtFrom.setText((str3 == null || str3.length() <= 0) ? this.countryName : this.countryName + ", " + this.cityName);
        }
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        int id = view.getId();
        if (id == R.id.title_img_back) {
            finish();
            return;
        }
        if (id == R.id.edit_layout_voiceSignature) {
            startActivity(new Intent(this, (Class<?>) VoiceSignatureActivity.class));
            return;
        }
        if (id == R.id.edit_txt_from) {
            Intent intent = new Intent(this, (Class<?>) CountryActivity.class);
            intent.putExtra(Constans.resultCountryId, this.countryId);
            intent.putExtra(Constans.resultCityId, this.cityId);
            intent.putExtra(Constans.resultCountryName, this.countryName);
            intent.putExtra(Constans.resultCityName, this.cityName);
            startActivityForResult(intent, 111);
            return;
        }
        if (id == R.id.edit_edit_birthday) {
            showDatePickerDialog();
            return;
        }
        if (id == R.id.edit_edit_email) {
            startActivityForResult(new Intent(this, (Class<?>) ResetEmailActivity.class), 222);
            return;
        }
        if (id == R.id.edit_txt_save) {
            setUserInfo();
            return;
        }
        if (id == R.id.edit_layout_bio) {
            startActivityForResult(new Intent(this, (Class<?>) EditBioActivity.class), 333);
            return;
        }
        if (id == R.id.txt_gender_female) {
            selectGender(0);
            return;
        }
        if (id == R.id.txt_gender_male) {
            selectGender(1);
            return;
        }
        if (id == R.id.txt_gender_trans) {
            selectGender(2);
            return;
        }
        if (id == R.id.txt_interested_female) {
            selectInterestedIn(0);
        } else if (id == R.id.txt_interested_male) {
            selectInterestedIn(1);
        } else if (id == R.id.txt_interested_trans) {
            selectInterestedIn(2);
        }
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, androidx.core.app.ComponentActivity, android.app.Activity
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        setContentView(R.layout.activity_details_edit);
        initView();
        init();
        setListener();
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.appcompat.app.AppCompatActivity, androidx.fragment.app.FragmentActivity, android.app.Activity
    protected void onDestroy() {
        super.onDestroy();
        unRegisterReceiver();
    }

    public void startPhotoZoom(Uri uri) {
        Intent intent = new Intent("com.android.camera.action.CROP");
        intent.setDataAndType(uri, SelectMimeType.SYSTEM_IMAGE);
        intent.putExtra("crop", ServerProtocol.DIALOG_RETURN_SCOPES_TRUE);
        intent.putExtra("aspectX", 1);
        intent.putExtra("aspectY", 1);
        intent.putExtra("outputX", 60);
        intent.putExtra("outputY", 60);
        intent.addFlags(3);
        intent.setClipData(ClipData.newRawUri("output", uri));
        this.uritempFile = uri;
        intent.putExtra("output", uri);
        intent.putExtra("return-data", false);
        intent.putExtra("outputFormat", Bitmap.CompressFormat.JPEG.toString());
        intent.putExtra("noFaceDetection", true);
        startActivityForResult(intent, 999);
    }
}
