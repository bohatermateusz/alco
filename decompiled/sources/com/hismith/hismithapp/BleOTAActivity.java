package com.hismith.hismithapp;

import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.fragment.app.FragmentActivity;
import com.bumptech.glide.Glide;
import com.google.android.material.timepicker.TimeModel;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.bluetooth.BleManager;
import com.hismith.hismithapp.dialog.OTADialogFragment;
import com.hismith.hismithapp.http.RetrofitRequest;
import com.hismith.hismithapp.http.RxSubscribe;
import com.hismith.hismithapp.model.BleDeviceModel;
import com.hismith.hismithapp.model.FirmwareModel;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.HexUtil;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import com.hismith.hismithapp.utils.ToastUtil;
import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.schedulers.Schedulers;

/* loaded from: classes3.dex */
public class BleOTAActivity extends BaseActivity implements View.OnClickListener {
    private RetrofitRequest baseRequest;
    private BleDeviceModel mBleDeviceModel;
    private Button mBtnCheck;
    private ImageView mImgLogo;
    private ImageView mTitleBack;
    private TextView mTitleName;
    private TextView mTxtName;
    private TextView mTxtNameLabel;
    private TextView mTxtNewVersion;
    private TextView mTxtNewVersionLabel;
    private TextView mTxtVersion;
    private TextView mTxtVersionLabel;
    private String newAppVersion;
    private String newVersion;

    private void firmwareVersion() {
        this.baseRequest.getApi().firmwareVersion((String) SharedPreferencesUtil.get(this, Constans.Token), "GV", MyApplication.getApplication().getDeviceCode()).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<FirmwareModel>(this, null, false) { // from class: com.hismith.hismithapp.BleOTAActivity.1
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(FirmwareModel firmwareModel) {
                byte[] hexStringToBytes;
                if (firmwareModel.getSuccess() == 1) {
                    BleOTAActivity.this.newVersion = firmwareModel.getFwVersion();
                    if (BleOTAActivity.this.newVersion == null || (hexStringToBytes = HexUtil.hexStringToBytes(BleOTAActivity.this.newVersion)) == null || hexStringToBytes.length < 4) {
                        return;
                    }
                    BleOTAActivity.this.newAppVersion = String.format("%02x", Byte.valueOf(hexStringToBytes[3])) + String.format("%02x", Byte.valueOf(hexStringToBytes[2]));
                    try {
                        int intValue = Integer.valueOf(BleOTAActivity.this.newAppVersion, 16).intValue();
                        BleOTAActivity bleOTAActivity = BleOTAActivity.this;
                        LanguageUtil.setText(bleOTAActivity, bleOTAActivity.mTxtNewVersion, String.format(TimeModel.ZERO_LEADING_NUMBER_FORMAT, Integer.valueOf(intValue)));
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
            }
        });
    }

    private void init() {
        try {
            BleDeviceModel connBleDevice = BleManager.getInstance(this).getBleProducts().get(getIntent().getStringExtra("otaBleAddress")).getConnBleDevice();
            this.mBleDeviceModel = connBleDevice;
            LanguageUtil.setText(this, this.mTxtName, connBleDevice.getDeviceName());
            String appVersion = this.mBleDeviceModel.getAppVersion();
            if (appVersion != null) {
                try {
                    LanguageUtil.setText(this, this.mTxtVersion, String.format(TimeModel.ZERO_LEADING_NUMBER_FORMAT, Integer.valueOf(Integer.valueOf(appVersion, 16).intValue())));
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        } catch (Exception e2) {
            e2.printStackTrace();
        }
        this.baseRequest = new RetrofitRequest();
        if (MyApplication.getApplication().getmProductModel() != null && MyApplication.getApplication().getmProductModel().getLogo() != null) {
            Glide.with((FragmentActivity) this).load(MyApplication.getApplication().getmProductModel().getLogo()).into(this.mImgLogo);
        }
        firmwareVersion();
    }

    private void initView() {
        this.mTitleBack = (ImageView) findViewById(R.id.title_img_back);
        this.mTitleName = (TextView) findViewById(R.id.title_txt_name);
        this.mImgLogo = (ImageView) findViewById(R.id.ota_img_logo);
        this.mTxtNameLabel = (TextView) findViewById(R.id.ota_txt_name_label);
        this.mTxtVersionLabel = (TextView) findViewById(R.id.ota_txt_version_label);
        this.mTxtNewVersionLabel = (TextView) findViewById(R.id.ota_txt_new_version_label);
        this.mTxtName = (TextView) findViewById(R.id.ota_txt_name);
        this.mTxtVersion = (TextView) findViewById(R.id.ota_txt_version);
        this.mTxtNewVersion = (TextView) findViewById(R.id.ota_txt_new_version);
        this.mBtnCheck = (Button) findViewById(R.id.ota_btn_check);
        LanguageUtil.setText(this, this.mTitleName, "firmwareUpgrade");
        LanguageUtil.setText(this, this.mTxtNameLabel, "deviceName");
        LanguageUtil.setText(this, this.mTxtVersionLabel, "firmwareVersion");
        LanguageUtil.setText(this, this.mTxtNewVersionLabel, "latestVersion");
        LanguageUtil.setText(this, this.mBtnCheck, "firmwareUpgrade");
    }

    private void setListener() {
        this.mTitleBack.setOnClickListener(this);
        this.mBtnCheck.setOnClickListener(this);
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        int id = view.getId();
        if (id == R.id.title_img_back) {
            finish();
            return;
        }
        if (id == R.id.ota_btn_check) {
            if (BleManager.getInstance(this).getStatus() != 2) {
                ToastUtil.showShortToastCenter(this, LanguageUtil.getLanguageByKey(this, "deviceDisconnected"));
                return;
            }
            if (this.mBleDeviceModel.getAppVersion() == null || this.newAppVersion == null) {
                ToastUtil.showShortToastCenter(this, LanguageUtil.getLanguageByKey(this, "noNeedUpgrade"));
            } else if (this.mBleDeviceModel.getAppVersion().compareTo(this.newAppVersion) == 0) {
                ToastUtil.showShortToastCenter(this, LanguageUtil.getLanguageByKey(this, "noNeedUpgrade"));
            } else {
                OTADialogFragment.getInstance(this.mBleDeviceModel.getAppVersion(), this.newAppVersion).show(getSupportFragmentManager(), "ota");
            }
        }
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, androidx.core.app.ComponentActivity, android.app.Activity
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        setContentView(R.layout.activity_ble_ota);
        initView();
        init();
        setListener();
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.appcompat.app.AppCompatActivity, androidx.fragment.app.FragmentActivity, android.app.Activity
    protected void onDestroy() {
        super.onDestroy();
    }
}
