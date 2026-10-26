package com.hismith.hismithapp.dialog;

import android.content.Context;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.text.TextUtils;
import android.util.Base64;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.TextView;
import androidx.fragment.app.DialogFragment;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.http.RetrofitRequest;
import com.hismith.hismithapp.http.RxSubscribe;
import com.hismith.hismithapp.listener.CustomClickListener;
import com.hismith.hismithapp.model.FirmwareModel;
import com.hismith.hismithapp.utils.Base64Util;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.HexUtil;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.MD5Util;
import com.hismith.hismithapp.utils.OrdUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import com.hismith.hismithapp.view.TextProgressBar;
import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.schedulers.Schedulers;

/* loaded from: classes3.dex */
public class OTADialogFragment extends DialogFragment {
    private static final int WHAT_COMPLETE = 1;
    private String appVersion;
    private RetrofitRequest baseRequest;
    private Button mBtnCancel;
    private Button mBtnConfirm;
    private Context mContext;
    private FrameLayout mLayoutConfirm;
    private LinearLayout mLayoutContent;
    private ProgressBar mProgressLoad;
    private TextProgressBar mProgressUpdate;
    private TextView mTxtContent;
    private TextView mTxtProgress;
    private TextView mTxtTitle;
    private String newAppVersion;
    private boolean isUpdate = false;
    private Handler mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.dialog.OTADialogFragment.1
        @Override // android.os.Handler.Callback
        public boolean handleMessage(Message message) {
            if (message.what == 1) {
                if (OTADialogFragment.this.mProgressUpdate != null) {
                    OTADialogFragment.this.mProgressUpdate.setVisibility(8);
                }
                if (OTADialogFragment.this.mTxtContent != null) {
                    LanguageUtil.setText(OTADialogFragment.this.mContext, OTADialogFragment.this.mTxtContent, "fUpgradeCompleted");
                }
                if (OTADialogFragment.this.mBtnCancel != null) {
                    LanguageUtil.setText(OTADialogFragment.this.mContext, OTADialogFragment.this.mBtnCancel, "done");
                    OTADialogFragment.this.mBtnCancel.setVisibility(0);
                }
                if (OTADialogFragment.this.mLayoutConfirm != null) {
                    OTADialogFragment.this.mLayoutConfirm.setVisibility(8);
                }
            }
            return false;
        }
    });

    /* JADX INFO: Access modifiers changed from: private */
    public void firmwareFile() {
        setBtnEnabled(this.mBtnConfirm, false);
        this.mTxtProgress.setVisibility(0);
        LanguageUtil.setText(this.mContext, this.mTxtProgress, "downloadingFirmware");
        this.mProgressLoad.setVisibility(0);
        this.mProgressUpdate.setVisibility(8);
        this.baseRequest.getApi().firmwareVersion((String) SharedPreferencesUtil.get(this.mContext, Constans.Token), "GF", MyApplication.getApplication().getDeviceCode()).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<FirmwareModel>(this.mContext, null, false) { // from class: com.hismith.hismithapp.dialog.OTADialogFragment.4
            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onComplete() {
                super.onComplete();
                OTADialogFragment.this.mProgressLoad.setVisibility(8);
            }

            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onError(Throwable th) {
                super.onError(th);
                LanguageUtil.setText(OTADialogFragment.this.mContext, OTADialogFragment.this.mTxtProgress, "downloadFailed");
                OTADialogFragment.this.mProgressLoad.setVisibility(8);
                OTADialogFragment oTADialogFragment = OTADialogFragment.this;
                oTADialogFragment.setBtnEnabled(oTADialogFragment.mBtnConfirm, true);
            }

            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(FirmwareModel firmwareModel) {
                int success = firmwareModel.getSuccess();
                OTADialogFragment oTADialogFragment = OTADialogFragment.this;
                if (success != 1) {
                    LanguageUtil.setText(oTADialogFragment.mContext, OTADialogFragment.this.mTxtProgress, "error 01");
                    OTADialogFragment oTADialogFragment2 = OTADialogFragment.this;
                    oTADialogFragment2.setBtnEnabled(oTADialogFragment2.mBtnConfirm, true);
                    return;
                }
                byte[] decodeFirmwareForByte = oTADialogFragment.decodeFirmwareForByte(firmwareModel.getFwPass(), firmwareModel.getFwData());
                if (decodeFirmwareForByte != null) {
                    LogUtil.e("decodeFirmware：" + HexUtil.bytesToHex(decodeFirmwareForByte));
                    return;
                }
                LanguageUtil.setText(OTADialogFragment.this.mContext, OTADialogFragment.this.mTxtProgress, "error 02");
                OTADialogFragment oTADialogFragment3 = OTADialogFragment.this;
                oTADialogFragment3.setBtnEnabled(oTADialogFragment3.mBtnConfirm, true);
            }
        });
    }

    public static OTADialogFragment getInstance(String str, String str2) {
        OTADialogFragment oTADialogFragment = new OTADialogFragment();
        Bundle bundle = new Bundle();
        bundle.putString("appVersion", str);
        bundle.putString("newAppVersion", str2);
        oTADialogFragment.setArguments(bundle);
        return oTADialogFragment;
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void setBtnEnabled(Button button, boolean z) {
        if (z) {
            button.setEnabled(true);
            button.setTextColor(getResources().getColor(R.color.white));
            button.setBackgroundResource(R.drawable.selector_btn);
        } else {
            button.setEnabled(false);
            button.setTextColor(getResources().getColor(R.color.gray9));
            button.setBackgroundResource(R.drawable.bg_edit_c57eba_5);
        }
    }

    private void setListener() {
        this.mBtnCancel.setOnClickListener(new CustomClickListener() { // from class: com.hismith.hismithapp.dialog.OTADialogFragment.2
            @Override // com.hismith.hismithapp.listener.CustomClickListener
            protected void onSingleClick(View view) {
                OTADialogFragment.this.dismiss();
            }
        });
        this.mBtnConfirm.setOnClickListener(new CustomClickListener() { // from class: com.hismith.hismithapp.dialog.OTADialogFragment.3
            @Override // com.hismith.hismithapp.listener.CustomClickListener
            protected void onSingleClick(View view) {
                if (OTADialogFragment.this.isUpdate) {
                    OTADialogFragment.this.firmwareFile();
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
        setStyle(2, R.style.otaDialogStyle);
        this.baseRequest = new RetrofitRequest();
    }

    @Override // androidx.fragment.app.Fragment
    public View onCreateView(LayoutInflater layoutInflater, ViewGroup viewGroup, Bundle bundle) {
        View inflate = layoutInflater.inflate(R.layout.dialog_ble_ota, viewGroup);
        this.mTxtTitle = (TextView) inflate.findViewById(R.id.dialog_ota_txt_title);
        this.mTxtContent = (TextView) inflate.findViewById(R.id.dialog_ota_txt_content);
        this.mTxtProgress = (TextView) inflate.findViewById(R.id.dialog_ota_txt_progress);
        this.mProgressLoad = (ProgressBar) inflate.findViewById(R.id.dialog_ota_progress_load);
        this.mProgressUpdate = (TextProgressBar) inflate.findViewById(R.id.dialog_ota_progress_update);
        this.mBtnCancel = (Button) inflate.findViewById(R.id.dialog_ota_btn_cancel);
        this.mBtnConfirm = (Button) inflate.findViewById(R.id.dialog_ota_btn_confirm);
        this.mLayoutConfirm = (FrameLayout) inflate.findViewById(R.id.dialog_ota_layout_confirm);
        setCancelable(false);
        this.mProgressUpdate.setProgress(0);
        this.mProgressUpdate.setMax(100);
        setListener();
        LanguageUtil.setText(this.mContext, this.mTxtTitle, "firmwareUpgrade");
        LanguageUtil.setText(this.mContext, this.mBtnCancel, "cancel");
        LanguageUtil.setText(this.mContext, this.mBtnConfirm, "upgrade");
        try {
            StringBuilder sb = new StringBuilder();
            Bundle arguments = getArguments();
            if (arguments != null) {
                this.appVersion = arguments.getString("appVersion");
                String string = arguments.getString("newAppVersion");
                this.newAppVersion = string;
                String str = this.appVersion;
                if (str != null && string != null) {
                    if (str.compareTo(string) == 0) {
                        sb.append("noNeedUpgrade");
                        this.isUpdate = false;
                    } else {
                        sb.append("noTurnDevice");
                        this.isUpdate = true;
                    }
                }
            }
            LanguageUtil.setText(this.mContext, this.mTxtContent, sb.toString());
        } catch (Exception e) {
            e.printStackTrace();
        }
        if (!this.isUpdate) {
            setBtnEnabled(this.mBtnConfirm, false);
        }
        return inflate;
    }

    @Override // androidx.fragment.app.Fragment
    public void onDestroy() {
        super.onDestroy();
    }

    @Override // androidx.fragment.app.DialogFragment, androidx.fragment.app.Fragment
    public void onStart() {
        super.onStart();
        if (getDialog() == null || getDialog().getWindow() == null) {
            return;
        }
        Window window = getDialog().getWindow();
        WindowManager.LayoutParams attributes = window.getAttributes();
        attributes.width = (int) (this.mContext.getResources().getDisplayMetrics().widthPixels * 0.9d);
        window.setAttributes(attributes);
    }
}
