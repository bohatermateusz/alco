package com.hismith.hismithapp.dialog;

import android.content.Context;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.text.SpannableStringBuilder;
import android.text.TextUtils;
import android.text.style.ForegroundColorSpan;
import android.util.Base64;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.view.Window;
import android.view.WindowManager;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;
import androidx.fragment.app.DialogFragment;
import com.bumptech.glide.Glide;
import com.google.android.material.timepicker.TimeModel;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.http.RetrofitRequest;
import com.hismith.hismithapp.http.RxSubscribe;
import com.hismith.hismithapp.listener.CustomClickListener;
import com.hismith.hismithapp.model.BleDeviceModel;
import com.hismith.hismithapp.model.FirmwareModel;
import com.hismith.hismithapp.utils.Base64Util;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.HexUtil;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.MD5Util;
import com.hismith.hismithapp.utils.OrdUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import com.hismith.hismithapp.view.CircleImageView;
import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.schedulers.Schedulers;

/* loaded from: classes3.dex */
public class OTADialogFragment2 extends DialogFragment {
    private static final int WHAT_COMPLETE = 1;
    private static final int WHAT_OLD_VERSION = 2;
    private static final int WHAT_PROGRESS = 3;
    private String appVersion;
    private RetrofitRequest baseRequest;
    private BleDeviceModel mBleDeviceModel;
    private Context mContext;
    private ImageView mImgBg;
    private ImageView mImgCancel;
    private CircleImageView mImgLogo;
    private FrameLayout mLayoutUpgrade;
    private ProgressBar mProgressUpgrade;
    private TextView mTxtError;
    private TextView mTxtName;
    private TextView mTxtNew;
    private TextView mTxtNewLabel;
    private TextView mTxtNote;
    private TextView mTxtOld;
    private TextView mTxtOldLabel;
    private TextView mTxtUpgrade;
    private String newAppVersion;
    private ProgressDialog progressDialog;
    private boolean isUpdate = false;
    private Handler mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.dialog.OTADialogFragment2.1
        @Override // android.os.Handler.Callback
        public boolean handleMessage(Message message) {
            int i = message.what;
            if (i == 1) {
                LanguageUtil.setText(OTADialogFragment2.this.mContext, OTADialogFragment2.this.mTxtUpgrade, "fUpgradeCompleted");
                OTADialogFragment2.this.setCancelable(true);
                return false;
            }
            if (i != 2) {
                if (i != 3) {
                    return false;
                }
                int intValue = ((Integer) message.obj).intValue();
                OTADialogFragment2.this.mProgressUpgrade.setProgress(intValue);
                if (intValue != 100) {
                    return false;
                }
                OTADialogFragment2.this.mHandler.sendEmptyMessage(1);
                return false;
            }
            OTADialogFragment2 oTADialogFragment2 = OTADialogFragment2.this;
            oTADialogFragment2.appVersion = oTADialogFragment2.mBleDeviceModel.getAppVersion();
            if (OTADialogFragment2.this.appVersion != null) {
                try {
                    LanguageUtil.setText(OTADialogFragment2.this.mContext, OTADialogFragment2.this.mTxtOld, String.format(TimeModel.ZERO_LEADING_NUMBER_FORMAT, Integer.valueOf(Integer.valueOf(OTADialogFragment2.this.appVersion, 16).intValue())));
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
            OTADialogFragment2.this.checkIsUpdate();
            return false;
        }
    });

    /* JADX INFO: Access modifiers changed from: private */
    public void checkIsUpdate() {
        String str;
        String str2 = this.appVersion;
        boolean z = false;
        if (str2 == null || (str = this.newAppVersion) == null) {
            this.isUpdate = false;
        } else if (str2.compareTo(str) == 0) {
            this.isUpdate = false;
        } else {
            z = true;
            this.isUpdate = true;
        }
        setBtnEnabled(z);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void firmwareFile() {
        this.mLayoutUpgrade.setEnabled(false);
        this.mTxtError.setText("");
        this.mImgBg.setVisibility(8);
        this.mProgressUpgrade.setProgress(0);
        this.mProgressUpgrade.setVisibility(0);
        this.baseRequest.getApi().firmwareVersion((String) SharedPreferencesUtil.get(this.mContext, Constans.Token), "GF", MyApplication.getApplication().getDeviceCode()).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<FirmwareModel>(this.mContext, this.progressDialog, true) { // from class: com.hismith.hismithapp.dialog.OTADialogFragment2.5
            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onComplete() {
                super.onComplete();
            }

            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onError(Throwable th) {
                super.onError(th);
                LanguageUtil.setText(OTADialogFragment2.this.mContext, OTADialogFragment2.this.mTxtError, "downloadFailed");
                OTADialogFragment2.this.mImgBg.setVisibility(0);
                OTADialogFragment2.this.mProgressUpgrade.setVisibility(8);
                OTADialogFragment2.this.setBtnEnabled(true);
            }

            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(FirmwareModel firmwareModel) {
                int success = firmwareModel.getSuccess();
                OTADialogFragment2 oTADialogFragment2 = OTADialogFragment2.this;
                if (success != 1) {
                    LanguageUtil.setText(oTADialogFragment2.mContext, OTADialogFragment2.this.mTxtError, "error 01");
                    OTADialogFragment2.this.setBtnEnabled(true);
                    return;
                }
                byte[] decodeFirmwareForByte = oTADialogFragment2.decodeFirmwareForByte(firmwareModel.getFwPass(), firmwareModel.getFwData());
                if (decodeFirmwareForByte != null) {
                    LogUtil.e("decodeFirmware：" + HexUtil.bytesToHex(decodeFirmwareForByte));
                } else {
                    LanguageUtil.setText(OTADialogFragment2.this.mContext, OTADialogFragment2.this.mTxtError, "error 02");
                    OTADialogFragment2.this.setBtnEnabled(true);
                }
            }
        });
    }

    private void firmwareVersion() {
        this.baseRequest.getApi().firmwareVersion((String) SharedPreferencesUtil.get(this.mContext, Constans.Token), "GV", MyApplication.getApplication().getDeviceCode()).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<FirmwareModel>(this.mContext, null, false) { // from class: com.hismith.hismithapp.dialog.OTADialogFragment2.4
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(FirmwareModel firmwareModel) {
                byte[] hexStringToBytes;
                if (firmwareModel.getSuccess() == 1) {
                    String fwVersion = firmwareModel.getFwVersion();
                    if (fwVersion != null && (hexStringToBytes = HexUtil.hexStringToBytes(fwVersion)) != null && hexStringToBytes.length >= 4) {
                        OTADialogFragment2.this.newAppVersion = String.format("%02x", Byte.valueOf(hexStringToBytes[3])) + String.format("%02x", Byte.valueOf(hexStringToBytes[2]));
                        try {
                            LanguageUtil.setText(OTADialogFragment2.this.mContext, OTADialogFragment2.this.mTxtNew, String.format(TimeModel.ZERO_LEADING_NUMBER_FORMAT, Integer.valueOf(Integer.valueOf(OTADialogFragment2.this.newAppVersion, 16).intValue())));
                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                    }
                    OTADialogFragment2.this.checkIsUpdate();
                }
            }
        });
    }

    public static OTADialogFragment2 getInstance(String str, String str2) {
        OTADialogFragment2 oTADialogFragment2 = new OTADialogFragment2();
        Bundle bundle = new Bundle();
        bundle.putString("appVersion", str);
        bundle.putString("newAppVersion", str2);
        oTADialogFragment2.setArguments(bundle);
        return oTADialogFragment2;
    }

    private void init() {
        this.baseRequest = new RetrofitRequest();
        this.progressDialog = new ProgressDialog(this.mContext, R.style.MyDialog);
        if (MyApplication.getApplication().getmProductModel() != null && MyApplication.getApplication().getmProductModel().getLogo() != null) {
            Glide.with(this).load(MyApplication.getApplication().getmProductModel().getLogo()).into(this.mImgLogo);
        }
        LanguageUtil.setText(this.mContext, this.mTxtName, this.mBleDeviceModel.getDeviceName());
        String appVersion = this.mBleDeviceModel.getAppVersion();
        this.appVersion = appVersion;
        if (appVersion != null) {
            try {
                LanguageUtil.setText(this.mContext, this.mTxtOld, String.format(TimeModel.ZERO_LEADING_NUMBER_FORMAT, Integer.valueOf(Integer.valueOf(appVersion, 16).intValue())));
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        checkIsUpdate();
        firmwareVersion();
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void setBtnEnabled(boolean z) {
        if (isAdded()) {
            FrameLayout frameLayout = this.mLayoutUpgrade;
            if (z) {
                frameLayout.setEnabled(true);
                this.mTxtUpgrade.setTextColor(getResources().getColor(R.color.white));
                this.mImgBg.setBackgroundResource(R.drawable.bg_corners_style_25);
            } else {
                frameLayout.setEnabled(false);
                this.mTxtUpgrade.setTextColor(getResources().getColor(R.color.gray9));
                this.mImgBg.setBackgroundResource(R.drawable.bg_corners_713367_25);
            }
        }
    }

    private void setListener() {
        this.mImgCancel.setOnClickListener(new CustomClickListener() { // from class: com.hismith.hismithapp.dialog.OTADialogFragment2.2
            @Override // com.hismith.hismithapp.listener.CustomClickListener
            protected void onSingleClick(View view) {
                OTADialogFragment2.this.dismiss();
            }
        });
        this.mLayoutUpgrade.setOnClickListener(new CustomClickListener() { // from class: com.hismith.hismithapp.dialog.OTADialogFragment2.3
            @Override // com.hismith.hismithapp.listener.CustomClickListener
            protected void onSingleClick(View view) {
                if (OTADialogFragment2.this.isUpdate) {
                    OTADialogFragment2.this.firmwareFile();
                }
            }
        });
    }

    public byte[] decodeFirmwareForByte(String str, String str2) {
        if (TextUtils.isEmpty(str2)) {
            return null;
        }
        String fromBase64 = Base64Util.getFromBase64(str2);
        int length = str.length();
        if (length > 32) {
            length = 32;
        }
        String string2MD5 = MD5Util.string2MD5(str);
        for (int i = 0; i < length; i++) {
            int i2 = length - i;
            String substring = string2MD5.substring(i2 - 1, i2);
            int ord = OrdUtil.ord(substring) % (fromBase64.length() - 1);
            int i3 = ord + 1;
            if (!fromBase64.substring(ord, i3).equals(substring)) {
                return null;
            }
            fromBase64 = fromBase64.substring(0, ord) + fromBase64.substring(i3);
        }
        return Base64.decode(fromBase64, 2);
    }

    public String decodeFirmwareForString(String str, String str2) {
        if (TextUtils.isEmpty(str2)) {
            return null;
        }
        String fromBase64 = Base64Util.getFromBase64(str2);
        int length = str.length();
        if (length > 32) {
            length = 32;
        }
        String string2MD5 = MD5Util.string2MD5(str);
        for (int i = 0; i < length; i++) {
            int i2 = length - i;
            String substring = string2MD5.substring(i2 - 1, i2);
            int ord = OrdUtil.ord(substring) % (fromBase64.length() - 1);
            int i3 = ord + 1;
            if (!fromBase64.substring(ord, i3).equals(substring)) {
                return null;
            }
            fromBase64 = fromBase64.substring(0, ord) + fromBase64.substring(i3);
        }
        return Base64Util.getFromBase64(fromBase64);
    }

    @Override // androidx.fragment.app.DialogFragment, androidx.fragment.app.Fragment
    public void onAttach(Context context) {
        super.onAttach(context);
        this.mContext = context;
    }

    @Override // androidx.fragment.app.DialogFragment, androidx.fragment.app.Fragment
    public void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        LogUtil.e("onCreate()");
        setStyle(2, R.style.otaDialogStyle);
    }

    @Override // androidx.fragment.app.Fragment
    public View onCreateView(LayoutInflater layoutInflater, ViewGroup viewGroup, Bundle bundle) {
        LogUtil.e("onCreateView()");
        View inflate = layoutInflater.inflate(R.layout.dialog_ble_ota2, viewGroup);
        this.mImgLogo = (CircleImageView) inflate.findViewById(R.id.dialog_ota_img_logo);
        this.mTxtName = (TextView) inflate.findViewById(R.id.dialog_ota_txt_name);
        this.mTxtOldLabel = (TextView) inflate.findViewById(R.id.dialog_ota_txt_old_label);
        this.mTxtOld = (TextView) inflate.findViewById(R.id.dialog_ota_txt_old);
        this.mTxtNewLabel = (TextView) inflate.findViewById(R.id.dialog_ota_txt_new_label);
        this.mTxtNew = (TextView) inflate.findViewById(R.id.dialog_ota_txt_new);
        this.mTxtError = (TextView) inflate.findViewById(R.id.dialog_ota_txt_error);
        this.mTxtUpgrade = (TextView) inflate.findViewById(R.id.dialog_ota_txt_upgrade);
        this.mTxtNote = (TextView) inflate.findViewById(R.id.dialog_ota_txt_note);
        this.mLayoutUpgrade = (FrameLayout) inflate.findViewById(R.id.dialog_ota_layout_upgrade);
        this.mProgressUpgrade = (ProgressBar) inflate.findViewById(R.id.dialog_ota_progress_upgrade);
        this.mImgCancel = (ImageView) inflate.findViewById(R.id.dialog_ota_img_cancel);
        this.mImgBg = (ImageView) inflate.findViewById(R.id.dialog_ota_img_bg);
        String languageByKey = LanguageUtil.getLanguageByKey(this.mContext, "noTurnDevice");
        SpannableStringBuilder spannableStringBuilder = new SpannableStringBuilder(languageByKey);
        int indexOf = languageByKey.indexOf(":");
        if (indexOf <= 0) {
            indexOf = languageByKey.indexOf("：");
        }
        spannableStringBuilder.setSpan(new ForegroundColorSpan(-1), 0, indexOf + 1, 18);
        this.mTxtNote.setText(spannableStringBuilder);
        LanguageUtil.setText(this.mContext, this.mTxtOldLabel, "firmwareVersion");
        LanguageUtil.setText(this.mContext, this.mTxtNewLabel, "latestVersion");
        LanguageUtil.setText(this.mContext, this.mTxtUpgrade, "firmwareUpgrade");
        setCancelable(true);
        init();
        setListener();
        return inflate;
    }

    @Override // androidx.fragment.app.Fragment
    public void onDestroy() {
        super.onDestroy();
        ProgressDialog progressDialog = this.progressDialog;
        if (progressDialog != null) {
            progressDialog.dismiss();
            this.progressDialog = null;
        }
    }

    @Override // androidx.fragment.app.DialogFragment, androidx.fragment.app.Fragment
    public void onStart() {
        super.onStart();
        LogUtil.e("onStart()");
        if (getDialog() == null || getDialog().getWindow() == null) {
            return;
        }
        Window window = getDialog().getWindow();
        WindowManager.LayoutParams attributes = window.getAttributes();
        attributes.width = (int) (this.mContext.getResources().getDisplayMetrics().widthPixels * 0.9d);
        window.setAttributes(attributes);
    }
}
