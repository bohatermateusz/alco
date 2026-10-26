package com.hismith.hismithapp;

import android.content.Intent;
import android.os.Bundle;
import android.provider.MediaStore;
import android.text.TextUtils;
import android.view.MotionEvent;
import android.view.View;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.core.content.ContextCompat;
import cn.bingoogolapple.qrcode.core.QRCodeView;
import cn.bingoogolapple.qrcode.zxing.ZXingView;
import com.hismith.hismithapp.http.HttpModel;
import com.hismith.hismithapp.http.RetrofitRequest;
import com.hismith.hismithapp.http.RxSubscribe;
import com.hismith.hismithapp.manager.BeepManager;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.PermissionUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import com.hismith.hismithapp.utils.StatusBarUtil;
import com.hismith.hismithapp.utils.ToastUtil;
import com.hismith.hismithapp.utils.UriUtils;
import com.luck.picture.lib.config.SelectMimeType;
import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.schedulers.Schedulers;
import java.util.regex.Pattern;

/* loaded from: classes3.dex */
public class QRCodeActivity extends BaseActivity implements View.OnClickListener, QRCodeView.Delegate {
    public static final int REQUEST_CODE_PHOTO = 2;
    private RetrofitRequest baseRequest;
    private BeepManager beepManager;
    private boolean isFlashlight = false;
    private ImageView mImgLight;
    private ImageView mImgPic;
    private LinearLayout mLayoutTitle;
    private ImageView mTitleImgBack;
    private TextView mTitleTxtName;
    private ZXingView mZXingView;
    private PermissionUtil permissionUtil;

    private void changeFlashlight() {
        boolean z = this.isFlashlight;
        this.isFlashlight = !z;
        ImageView imageView = this.mImgLight;
        if (z) {
            imageView.setImageDrawable(ContextCompat.getDrawable(this, R.mipmap.ic_scan_light_close));
            this.mZXingView.closeFlashlight();
        } else {
            imageView.setImageDrawable(ContextCompat.getDrawable(this, R.mipmap.ic_scan_light_open));
            this.mZXingView.openFlashlight();
        }
    }

    private void init() {
        this.baseRequest = new RetrofitRequest();
        this.permissionUtil = new PermissionUtil(this);
        this.beepManager = new BeepManager(this);
    }

    private void initView() {
        this.mLayoutTitle = (LinearLayout) findViewById(R.id.layout_title);
        this.mTitleImgBack = (ImageView) findViewById(R.id.title_img_back);
        this.mTitleTxtName = (TextView) findViewById(R.id.title_txt_name);
        ZXingView zXingView = (ZXingView) findViewById(R.id.zxingview);
        this.mZXingView = zXingView;
        zXingView.setDelegate(this);
        this.mImgPic = (ImageView) findViewById(R.id.scan_img_pic);
        this.mImgLight = (ImageView) findViewById(R.id.scan_img_light);
        LanguageUtil.setText(this, this.mTitleTxtName, "scan");
        StatusBarUtil.fitTitleBar(this, this.mLayoutTitle);
    }

    private boolean isHttpUrl(String str) {
        if (TextUtils.isEmpty(str)) {
            return false;
        }
        boolean matches = Pattern.compile("(((https|http)?://)?([a-z0-9]+[.])|(www.))\\w+[.|\\/]([a-z0-9]{0,})?[[.]([a-z0-9]{0,})]+((/[\\S&&[^,;一-龥]]+)+)?([.][a-z0-9]{0,}+|/?)".trim()).matcher(str.trim()).matches();
        if (matches) {
            return true;
        }
        return matches;
    }

    private void setListener() {
        this.mTitleImgBack.setOnClickListener(this);
        this.mImgPic.setOnClickListener(this);
        this.mImgLight.setOnClickListener(this);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void startPhotoCode() {
        Intent intent = new Intent("android.intent.action.PICK", MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        intent.setDataAndType(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, SelectMimeType.SYSTEM_IMAGE);
        startActivityForResult(intent, 2);
    }

    @Override // androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, android.app.Activity
    protected void onActivityResult(int i, int i2, Intent intent) {
        super.onActivityResult(i, i2, intent);
        this.mZXingView.showScanRect();
        if (i2 == -1 && i == 2) {
            String imagePath = UriUtils.getImagePath(this, intent);
            LogUtil.e("获取二维码图片：" + imagePath);
            if (TextUtils.isEmpty(imagePath)) {
                return;
            }
            this.mZXingView.decodeQRCode(imagePath);
        }
    }

    @Override // cn.bingoogolapple.qrcode.core.QRCodeView.Delegate
    public void onCameraAmbientBrightnessChanged(boolean z) {
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        int id = view.getId();
        if (id == R.id.title_img_back) {
            finish();
        } else if (id == R.id.scan_img_pic) {
            this.permissionUtil.initImagePermission(new PermissionUtil.OnGrantedListener() { // from class: com.hismith.hismithapp.QRCodeActivity.2
                @Override // com.hismith.hismithapp.utils.PermissionUtil.OnGrantedListener
                public void onGranted() {
                    QRCodeActivity.this.startPhotoCode();
                }
            });
        } else if (id == R.id.scan_img_light) {
            changeFlashlight();
        }
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, androidx.core.app.ComponentActivity, android.app.Activity
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        setContentView(R.layout.activity_qrcode);
        initView();
        init();
        setListener();
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.appcompat.app.AppCompatActivity, androidx.fragment.app.FragmentActivity, android.app.Activity
    protected void onDestroy() {
        this.mZXingView.onDestroy();
        super.onDestroy();
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, android.app.Activity
    protected void onPause() {
        super.onPause();
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, android.app.Activity
    protected void onResume() {
        super.onResume();
    }

    @Override // cn.bingoogolapple.qrcode.core.QRCodeView.Delegate
    public void onScanQRCodeOpenCameraError() {
        LogUtil.e("打开相机出错");
    }

    @Override // cn.bingoogolapple.qrcode.core.QRCodeView.Delegate
    public void onScanQRCodeSuccess(String str) {
        LogUtil.e("result:" + str);
        this.beepManager.playBeepSoundAndVibrate();
        if (TextUtils.isEmpty(str)) {
            this.mZXingView.startSpot();
            return;
        }
        if (str.contains("www.hismiths.com/qrcode.html")) {
            qrCodeLogin(str);
            return;
        }
        if (isHttpUrl(str)) {
            Intent intent = new Intent(this, (Class<?>) WebActivity.class);
            intent.putExtra("extra_web_url", str);
            startActivity(intent);
            finish();
            return;
        }
        Intent intent2 = new Intent(this, (Class<?>) TextActivity.class);
        intent2.putExtra(TextActivity.EXTRA_TEXT_CONTENT, str);
        startActivity(intent2);
        finish();
    }

    @Override // androidx.appcompat.app.AppCompatActivity, androidx.fragment.app.FragmentActivity, android.app.Activity
    protected void onStart() {
        super.onStart();
        this.mZXingView.startCamera();
        this.mZXingView.startSpotAndShowRect();
    }

    @Override // androidx.appcompat.app.AppCompatActivity, androidx.fragment.app.FragmentActivity, android.app.Activity
    protected void onStop() {
        this.mZXingView.stopCamera();
        super.onStop();
    }

    @Override // android.app.Activity
    public boolean onTouchEvent(MotionEvent motionEvent) {
        return super.onTouchEvent(motionEvent);
    }

    public void qrCodeLogin(String str) {
        this.baseRequest.getApi().qrcodeLogin(str, (String) SharedPreferencesUtil.get(this, Constans.Token)).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<HttpModel>(this, this.progressDialog, true) { // from class: com.hismith.hismithapp.QRCodeActivity.1
            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onComplete() {
                super.onComplete();
                QRCodeActivity.this.finish();
            }

            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onError(Throwable th) {
                super.onError(th);
                QRCodeActivity.this.mZXingView.startSpot();
            }

            @Override // com.hismith.hismithapp.http.RxSubscribe
            protected void onSuccess(HttpModel httpModel) {
                if (httpModel.getSuccess() != 0) {
                    ToastUtil.showShortToastCenter(QRCodeActivity.this, LanguageUtil.getLanguageByKey(QRCodeActivity.this, "loginSuccessful"));
                } else {
                    if (httpModel.getErrors() == null || httpModel.getErrors().length <= 0 || httpModel.getErrors()[0].getErrorMessage().length() <= 0) {
                        return;
                    }
                    Intent intent = new Intent(QRCodeActivity.this, (Class<?>) TextActivity.class);
                    intent.putExtra(TextActivity.EXTRA_TEXT_CONTENT, httpModel.getErrors()[0].getErrorMessage());
                    QRCodeActivity.this.startActivity(intent);
                }
            }
        });
    }
}
