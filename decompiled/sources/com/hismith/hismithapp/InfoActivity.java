package com.hismith.hismithapp;

import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.text.TextUtils;
import android.view.View;
import android.view.inputmethod.InputMethodManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.activity.OnBackPressedCallback;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.FragmentActivity;
import com.bumptech.glide.Glide;
import com.bumptech.glide.request.BaseRequestOptions;
import com.bumptech.glide.request.RequestOptions;
import com.facebook.internal.NativeProtocol;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.dialog.ChoosePhotoDialog;
import com.hismith.hismithapp.dialog.DatePickerDialog;
import com.hismith.hismithapp.glide.GlideRoundTransform;
import com.hismith.hismithapp.http.HttpModel;
import com.hismith.hismithapp.http.RetrofitRequest;
import com.hismith.hismithapp.http.RxSubscribe;
import com.hismith.hismithapp.manager.PictureResultManager;
import com.hismith.hismithapp.model.UploadModel;
import com.hismith.hismithapp.model.UserModel;
import com.hismith.hismithapp.picker.PictureSelectorManager;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.RequestBodyUtil;
import com.hismith.hismithapp.utils.ToastUtil;
import com.luck.picture.lib.config.PictureMimeType;
import com.luck.picture.lib.entity.LocalMedia;
import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.disposables.CompositeDisposable;
import io.reactivex.schedulers.Schedulers;
import java.io.File;
import java.util.Calendar;
import java.util.HashMap;
import java.util.List;

/* loaded from: classes3.dex */
public class InfoActivity extends BaseActivity implements View.OnClickListener {
    private RetrofitRequest baseRequest;
    private Calendar calendar;
    private ChoosePhotoDialog chooseDialog;
    private DatePickerDialog datePickerDialog;
    private RetrofitRequest fileRequest;
    private String fileUrl;
    private Uri imgUriOri;
    private Button mBtnSubmit;
    private CompositeDisposable mCompositeDisposable;
    private EditText mEditBirthdate;
    private EditText mEditCountry;
    private EditText mEditName;
    private EditText mEditPhone;
    private ImageView mImgHead;
    private ImageView mRbFemale;
    private ImageView mRbMale;
    private ImageView mRbTransgender;
    private ImageView mTitleImgBack;
    private TextView mTitleTxtName;
    private Calendar maxCalendar;
    private PictureResultManager pictureResultManager;
    private String uriPath;
    private int mGender = 0;
    private int countryId = 0;
    private int cityId = 0;
    private String countryName = "";
    private String cityName = "";
    private String token = "";
    private final int requestCountry = 111;
    private final int requestCamera = 777;
    private final int requestAlbum = 888;

    private void init() {
        if (getIntent() != null) {
            this.token = getIntent().getStringExtra(Constans.Token);
        }
        this.baseRequest = new RetrofitRequest(RetrofitRequest.EnumRequest.REQUEST_BASE);
        this.fileRequest = new RetrofitRequest(RetrofitRequest.EnumRequest.REQUEST_FILE);
        this.pictureResultManager = new PictureResultManager(this);
        this.mCompositeDisposable = new CompositeDisposable();
        Calendar calendar = Calendar.getInstance();
        this.maxCalendar = calendar;
        calendar.add(1, -18);
        Calendar calendar2 = Calendar.getInstance();
        this.calendar = calendar2;
        calendar2.add(1, -18);
        this.chooseDialog = new ChoosePhotoDialog(this) { // from class: com.hismith.hismithapp.InfoActivity.2
            @Override // com.hismith.hismithapp.dialog.ChoosePhotoDialog
            public void btnClickDelete() {
            }

            @Override // com.hismith.hismithapp.dialog.ChoosePhotoDialog
            public void btnClickPhoto() {
                InfoActivity.this.pictureResultManager.openGallery(new PictureResultManager.OnGalleryListener() { // from class: com.hismith.hismithapp.InfoActivity.2.2
                    @Override // com.hismith.hismithapp.manager.PictureResultManager.OnGalleryListener
                    public void onGalleryResult(Uri uri) {
                        if (uri != null) {
                            Glide.with((FragmentActivity) InfoActivity.this).load(uri).apply((BaseRequestOptions<?>) new RequestOptions().transform(new GlideRoundTransform(InfoActivity.this, 25))).into(InfoActivity.this.mImgHead);
                            InfoActivity.this.uriPath = uri.toString();
                        }
                    }

                    @Override // com.hismith.hismithapp.manager.PictureResultManager.OnGalleryListener
                    public void onGalleryResultFile(String str) {
                        if (TextUtils.isEmpty(str)) {
                            return;
                        }
                        Glide.with((FragmentActivity) InfoActivity.this).load(PictureMimeType.isContent(str) ? Uri.parse(str) : str).apply((BaseRequestOptions<?>) new RequestOptions().transform(new GlideRoundTransform(InfoActivity.this, 25))).into(InfoActivity.this.mImgHead);
                        InfoActivity.this.uriPath = str;
                    }
                });
            }

            @Override // com.hismith.hismithapp.dialog.ChoosePhotoDialog
            public void btnClickShoot() {
                PictureSelectorManager.getInstance(InfoActivity.this).openCamera(new PictureSelectorManager.OnCameraListener() { // from class: com.hismith.hismithapp.InfoActivity.2.1
                    @Override // com.hismith.hismithapp.picker.PictureSelectorManager.OnCameraListener
                    public void onCameraResult(List<LocalMedia> list) {
                        for (int i = 0; i < list.size(); i++) {
                            if (list.get(i).isCompressed()) {
                                String compressPath = list.get(i).getCompressPath();
                                LogUtil.e("新图片库拍摄压缩后：" + compressPath);
                                Glide.with((FragmentActivity) InfoActivity.this).load(PictureMimeType.isContent(compressPath) ? Uri.parse(compressPath) : compressPath).apply((BaseRequestOptions<?>) new RequestOptions().transform(new GlideRoundTransform(InfoActivity.this, 25))).into(InfoActivity.this.mImgHead);
                                InfoActivity.this.uriPath = compressPath;
                            } else {
                                String realPath = list.get(i).getRealPath();
                                Glide.with((FragmentActivity) InfoActivity.this).load(PictureMimeType.isContent(realPath) ? Uri.parse(realPath) : realPath).apply((BaseRequestOptions<?>) new RequestOptions().transform(new GlideRoundTransform(InfoActivity.this, 25))).into(InfoActivity.this.mImgHead);
                                InfoActivity.this.uriPath = realPath;
                            }
                        }
                    }
                });
            }
        };
        initUserInfo();
    }

    private void initUserInfo() {
        if (MyApplication.getApplication().getmUserModel() != null) {
            Glide.with((FragmentActivity) this).load(MyApplication.getApplication().getmUserModel().getHeadIcon()).placeholder(R.mipmap.ic_info_head).apply((BaseRequestOptions<?>) new RequestOptions().transform(new GlideRoundTransform(this, 25))).into(this.mImgHead);
            this.mEditName.setText(MyApplication.getApplication().getmUserModel().getName());
            selectedGender(MyApplication.getApplication().getmUserModel().getGender());
            this.cityId = MyApplication.getApplication().getmUserModel().getCityId();
            this.cityName = MyApplication.getApplication().getmUserModel().getCity();
            this.countryId = MyApplication.getApplication().getmUserModel().getCountryId();
            this.countryName = MyApplication.getApplication().getmUserModel().getCountry();
            boolean isEmpty = TextUtils.isEmpty(MyApplication.getApplication().getmUserModel().getCity());
            EditText editText = this.mEditCountry;
            if (isEmpty) {
                editText.setText(MyApplication.getApplication().getmUserModel().getCountry());
            } else {
                editText.setText(MyApplication.getApplication().getmUserModel().getCountry() + ", " + MyApplication.getApplication().getmUserModel().getCity());
            }
            this.mEditBirthdate.setText(MyApplication.getApplication().getmUserModel().getBirthday());
            this.mEditPhone.setText(MyApplication.getApplication().getmUserModel().getPhone());
        }
    }

    private void initView() {
        this.mTitleImgBack = (ImageView) findViewById(R.id.title_img_back);
        this.mTitleTxtName = (TextView) findViewById(R.id.title_txt_name);
        this.mImgHead = (ImageView) findViewById(R.id.info_img_head);
        this.mRbFemale = (ImageView) findViewById(R.id.info_rb_female);
        this.mRbMale = (ImageView) findViewById(R.id.info_rb_male);
        this.mRbTransgender = (ImageView) findViewById(R.id.info_rb_transgender);
        this.mEditName = (EditText) findViewById(R.id.info_edit_name);
        this.mEditCountry = (EditText) findViewById(R.id.info_edit_country);
        this.mEditBirthdate = (EditText) findViewById(R.id.info_edit_birthdate);
        this.mEditPhone = (EditText) findViewById(R.id.info_edit_phone);
        this.mBtnSubmit = (Button) findViewById(R.id.info_btn_submit);
        LanguageUtil.setText(this, this.mTitleTxtName, "perfectPersonalInformation");
        LanguageUtil.setHint(this, this.mEditName, "pleaseEnterNickname");
        LanguageUtil.setHint(this, this.mEditCountry, "pleaseSelectCountry");
        LanguageUtil.setHint(this, this.mEditBirthdate, "pleaseEnterBirthday");
        LanguageUtil.setHint(this, this.mEditPhone, "pleaseEnterPhone");
        LanguageUtil.setText(this, this.mBtnSubmit, "submit");
    }

    private void selectedGender(int i) {
        if (i == 0) {
            this.mGender = 0;
            this.mRbMale.setImageDrawable(ContextCompat.getDrawable(this, R.mipmap.ic_info_men_normal));
            this.mRbFemale.setImageDrawable(ContextCompat.getDrawable(this, R.mipmap.ic_info_women_select));
            this.mRbTransgender.setImageDrawable(ContextCompat.getDrawable(this, R.mipmap.ic_info_transgender_normal));
            return;
        }
        if (i == 1) {
            this.mGender = 1;
            this.mRbMale.setImageDrawable(ContextCompat.getDrawable(this, R.mipmap.ic_info_men_select));
            this.mRbFemale.setImageDrawable(ContextCompat.getDrawable(this, R.mipmap.ic_info_women_normal));
            this.mRbTransgender.setImageDrawable(ContextCompat.getDrawable(this, R.mipmap.ic_info_transgender_normal));
            return;
        }
        if (i == 2) {
            this.mGender = 2;
            this.mRbMale.setImageDrawable(ContextCompat.getDrawable(this, R.mipmap.ic_info_men_normal));
            this.mRbFemale.setImageDrawable(ContextCompat.getDrawable(this, R.mipmap.ic_info_women_normal));
            this.mRbTransgender.setImageDrawable(ContextCompat.getDrawable(this, R.mipmap.ic_info_transgender_select));
        }
    }

    private void setListener() {
        this.mTitleImgBack.setOnClickListener(this);
        this.mImgHead.setOnClickListener(this);
        this.mRbFemale.setOnClickListener(this);
        this.mRbMale.setOnClickListener(this);
        this.mRbTransgender.setOnClickListener(this);
        this.mEditCountry.setOnClickListener(this);
        this.mEditBirthdate.setOnClickListener(this);
        this.mBtnSubmit.setOnClickListener(this);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void setUserInfo() {
        final String trim = this.mEditName.getText().toString().trim();
        final String trim2 = this.mEditPhone.getText().toString().trim();
        final String obj = this.mEditBirthdate.getText().toString();
        this.baseRequest.getApi().setUserInfo(this.token, trim, Integer.valueOf(this.mGender), Integer.valueOf(this.countryId), Integer.valueOf(this.cityId), obj, trim2, this.fileUrl).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<HttpModel>(this, this.progressDialog) { // from class: com.hismith.hismithapp.InfoActivity.5
            @Override // com.hismith.hismithapp.http.RxSubscribe
            protected void onSuccess(HttpModel httpModel) {
                if (httpModel.getSuccess() == 1) {
                    ToastUtil.showShortToastCenter(InfoActivity.this, R.string.android_saveed);
                    UserModel userModel = MyApplication.getApplication().getmUserModel();
                    if (userModel != null) {
                        userModel.setName(trim);
                        userModel.setGender(InfoActivity.this.mGender);
                        userModel.setBirthday(obj);
                        userModel.setPhone(trim2);
                        userModel.setCountry(InfoActivity.this.countryName);
                        userModel.setCountryId(InfoActivity.this.countryId);
                        userModel.setCity(InfoActivity.this.cityName);
                        userModel.setCityId(InfoActivity.this.cityId);
                        MyApplication.getApplication().setmUserModel(userModel);
                        InfoActivity.this.sendBroadcast(new Intent(Constans.RECEIVER_USER_INFO_CHANGE));
                    }
                    InfoActivity.this.finish();
                }
            }
        });
    }

    private void uploadFile() {
        if (TextUtils.isEmpty(this.uriPath)) {
            setUserInfo();
            return;
        }
        HashMap hashMap = new HashMap();
        RequestBodyUtil.addMultiPart(this, hashMap, Constans.Token, this.token);
        RequestBodyUtil.addMultiPart(this, hashMap, NativeProtocol.WEB_DIALOG_ACTION, "upload");
        RequestBodyUtil.addMultiPart(this, hashMap, "fileType", "image");
        if (PictureMimeType.isContent(this.uriPath)) {
            Uri parse = Uri.parse(this.uriPath);
            RequestBodyUtil.addMultiPartUri(this, hashMap, RequestBodyUtil.getFileNameFromUri(this, parse), parse);
        } else {
            File file = new File(this.uriPath);
            if (file.exists()) {
                RequestBodyUtil.addMultiPart(this, hashMap, "file", file);
            } else {
                setUserInfo();
            }
        }
        this.fileRequest.getApi().uploadFile(hashMap).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<UploadModel>(this, this.progressDialog) { // from class: com.hismith.hismithapp.InfoActivity.4
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(UploadModel uploadModel) {
                if (uploadModel.getSuccess() == 1) {
                    InfoActivity.this.fileUrl = uploadModel.getFileUrl();
                    InfoActivity.this.setUserInfo();
                }
            }
        });
    }

    public void hideSoftKeyboard() {
        View currentFocus = getCurrentFocus();
        if (currentFocus != null) {
            ((InputMethodManager) getSystemService("input_method")).hideSoftInputFromWindow(currentFocus.getWindowToken(), 2);
        }
    }

    @Override // androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, android.app.Activity
    protected void onActivityResult(int i, int i2, Intent intent) {
        super.onActivityResult(i, i2, intent);
        if (i == 111 && intent != null) {
            this.countryId = intent.getIntExtra(Constans.resultCountryId, 0);
            this.cityId = intent.getIntExtra(Constans.resultCityId, 0);
            this.countryName = intent.getStringExtra(Constans.resultCountryName);
            String stringExtra = intent.getStringExtra(Constans.resultCityName);
            this.cityName = stringExtra;
            this.mEditCountry.setText((stringExtra == null || stringExtra.length() <= 0) ? this.countryName : this.countryName + ", " + this.cityName);
        }
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        hideSoftKeyboard();
        int id = view.getId();
        if (id == R.id.title_img_back) {
            if (MyApplication.getApplication().getmUserModel() == null || TextUtils.isEmpty(MyApplication.getApplication().getmUserModel().getName())) {
                ToastUtil.showShortToast(this, LanguageUtil.getLanguageByKey(this, "pleaseEnterNickname"));
                return;
            } else {
                this.appManager.finishActivity();
                return;
            }
        }
        if (id == R.id.info_img_head) {
            this.chooseDialog.show();
            return;
        }
        if (id == R.id.info_rb_female) {
            selectedGender(0);
            return;
        }
        if (id == R.id.info_rb_male) {
            selectedGender(1);
            return;
        }
        if (id == R.id.info_rb_transgender) {
            selectedGender(2);
            return;
        }
        if (id == R.id.info_edit_country) {
            Intent intent = new Intent(this, (Class<?>) CountryActivity.class);
            intent.putExtra(Constans.resultCountryId, this.countryId);
            intent.putExtra(Constans.resultCityId, this.cityId);
            intent.putExtra(Constans.resultCountryName, this.countryName);
            intent.putExtra(Constans.resultCityName, this.cityName);
            startActivityForResult(intent, 111);
            return;
        }
        if (id == R.id.info_edit_birthdate) {
            showDatePickerDialog();
        } else if (id == R.id.info_btn_submit) {
            uploadFile();
        }
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, androidx.core.app.ComponentActivity, android.app.Activity
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        getOnBackPressedDispatcher().addCallback(this, new OnBackPressedCallback(true) { // from class: com.hismith.hismithapp.InfoActivity.1
            @Override // androidx.activity.OnBackPressedCallback
            public void handleOnBackPressed() {
                if (MyApplication.getApplication().getmUserModel() != null && !TextUtils.isEmpty(MyApplication.getApplication().getmUserModel().getName())) {
                    InfoActivity.this.finish();
                } else {
                    InfoActivity infoActivity = InfoActivity.this;
                    ToastUtil.showShortToast(infoActivity, LanguageUtil.getLanguageByKey(infoActivity, "pleaseEnterNickname"));
                }
            }
        });
        setContentView(R.layout.activity_info);
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
    }

    public void showDatePickerDialog() {
        DatePickerDialog datePickerDialog = this.datePickerDialog;
        if (datePickerDialog != null) {
            datePickerDialog.dismiss();
        }
        DatePickerDialog datePickerDialog2 = new DatePickerDialog(this, this.calendar.get(1), this.calendar.get(2), this.calendar.get(5)) { // from class: com.hismith.hismithapp.InfoActivity.3
            @Override // com.hismith.hismithapp.dialog.DatePickerDialog
            public void btnSubmit(int i, int i2, int i3) {
                InfoActivity.this.mEditBirthdate.setText(i + "-" + (i2 < 10 ? new StringBuilder("0") : new StringBuilder("")).append(i2).toString() + "-" + (i3 < 10 ? new StringBuilder("0").append(i3) : new StringBuilder("").append(i3)).toString());
                InfoActivity.this.calendar.set(i, i2 - 1, i3);
            }
        };
        this.datePickerDialog = datePickerDialog2;
        datePickerDialog2.showDialog();
    }
}
