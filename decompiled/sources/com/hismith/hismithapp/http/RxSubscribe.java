package com.hismith.hismithapp.http;

import android.content.Context;
import android.content.Intent;
import com.hismith.hismithapp.BaseActivity;
import com.hismith.hismithapp.LoginActivity;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.dialog.ProgressDialog;
import com.hismith.hismithapp.http.HttpModel;
import com.hismith.hismithapp.utils.AppManager;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import com.hismith.hismithapp.utils.ToastUtil;
import io.reactivex.Observer;
import io.reactivex.disposables.Disposable;

/* loaded from: classes3.dex */
public abstract class RxSubscribe<T extends HttpModel> implements Observer<T> {
    private boolean isNetConnect;
    private boolean isProgress;
    private Context mContext;
    private ProgressDialog progressDialog;

    public RxSubscribe(Context context, ProgressDialog progressDialog) {
        this.isProgress = true;
        this.isNetConnect = true;
        this.mContext = context;
        this.progressDialog = progressDialog;
    }

    public RxSubscribe(Context context, ProgressDialog progressDialog, boolean z) {
        this.isNetConnect = true;
        this.mContext = context;
        this.isProgress = z;
        this.progressDialog = progressDialog;
    }

    public RxSubscribe(Context context, ProgressDialog progressDialog, boolean z, boolean z2) {
        this.mContext = context;
        this.isProgress = z;
        this.progressDialog = progressDialog;
        this.isNetConnect = z2;
    }

    @Override // io.reactivex.Observer
    public void onComplete() {
        ProgressDialog progressDialog;
        if (this.isProgress && (progressDialog = this.progressDialog) != null && progressDialog.isShowing()) {
            this.progressDialog.dismiss();
        }
    }

    @Override // io.reactivex.Observer
    public void onError(Throwable th) {
        ProgressDialog progressDialog;
        LogUtil.e("请求失败信息: " + th.getMessage());
        if (this.isProgress) {
            ToastUtil.showShortToastCenter(this.mContext, LanguageUtil.getLanguageByKey(this.mContext, "networkConnectionTimedOut"));
        }
        if (this.isProgress && (progressDialog = this.progressDialog) != null && progressDialog.isShowing()) {
            this.progressDialog.dismiss();
        }
    }

    @Override // io.reactivex.Observer
    public void onNext(T t) {
        LogUtil.w("请求成功信息: " + t.toString());
        if (t.getSuccess() == 0 && t.getErrors() != null && t.getErrors().length > 0) {
            String errorMessage = t.getErrors()[0].getErrorMessage().length() > 0 ? t.getErrors()[0].getErrorMessage() : "";
            LogUtil.e("请求成功错误信息: " + t.getErrors()[0].getErrorCode() + " ,ErrorMessage:" + errorMessage);
            if (!"requestLimit".equals(t.getErrors()[0].getErrorCode()) && !"server error".equals(t.getErrors()[0].getErrorCode())) {
                Context context = this.mContext;
                if (!(context instanceof BaseActivity)) {
                    String str = errorMessage;
                    if (context instanceof MyApplication) {
                        ToastUtil.showShortToastCenter(MyApplication.getApplication(), str);
                    }
                } else if (Constans.errorRelogin.equals(t.getErrors()[0].getErrorCode())) {
                    String str2 = (String) SharedPreferencesUtil.get(this.mContext, Constans.Language);
                    Object obj = SharedPreferencesUtil.get(this.mContext, Constans.contactUserId, 0);
                    int intValue = obj != null ? ((Integer) obj).intValue() : 0;
                    String str3 = (String) SharedPreferencesUtil.get(this.mContext, Constans.storeUrl);
                    String str4 = (String) SharedPreferencesUtil.get(this.mContext, Constans.roomVoice);
                    String str5 = (String) SharedPreferencesUtil.get(this.mContext, Constans.tempToken);
                    String str6 = (String) SharedPreferencesUtil.get(this.mContext, Constans.aboutApp);
                    String str7 = (String) SharedPreferencesUtil.get(this.mContext, Constans.copyright);
                    int i = intValue;
                    String str8 = errorMessage;
                    String str9 = (String) SharedPreferencesUtil.get(this.mContext, Constans.mainBackgroundUrl);
                    String str10 = (String) SharedPreferencesUtil.get(this.mContext, Constans.bluetoothConnAddress);
                    String str11 = (String) SharedPreferencesUtil.get(this.mContext, Constans.ProductAllData);
                    MyApplication.getApplication().setmUserModel(null);
                    SharedPreferencesUtil.clear(this.mContext);
                    SharedPreferencesUtil.put(this.mContext, Constans.Language, str2);
                    SharedPreferencesUtil.put(this.mContext, Constans.contactUserId, Integer.valueOf(i));
                    SharedPreferencesUtil.put(this.mContext, Constans.storeUrl, str3);
                    SharedPreferencesUtil.put(this.mContext, Constans.roomVoice, str4);
                    SharedPreferencesUtil.put(this.mContext, Constans.tempToken, str5);
                    SharedPreferencesUtil.put(this.mContext, Constans.aboutApp, str6);
                    SharedPreferencesUtil.put(this.mContext, Constans.copyright, str7);
                    SharedPreferencesUtil.put(this.mContext, Constans.mainBackgroundUrl, str9);
                    SharedPreferencesUtil.put(this.mContext, Constans.bluetoothConnAddress, str10);
                    SharedPreferencesUtil.put(this.mContext, Constans.ProductAllData, str11);
                    try {
                        if (AppManager.getInstance().currentActivity() != null && !"LoginActivity".equals(AppManager.getInstance().currentActivity().getLocalClassName())) {
                            ToastUtil.showShortToastCenter(MyApplication.getApplication(), str8);
                            this.mContext.startActivity(new Intent(this.mContext, (Class<?>) LoginActivity.class));
                            ((BaseActivity) this.mContext).appManager.finishAllOtherActivity(LoginActivity.class);
                        }
                    } catch (Exception e) {
                        e.printStackTrace();
                        this.mContext.startActivity(new Intent(this.mContext, (Class<?>) LoginActivity.class));
                        ((BaseActivity) this.mContext).appManager.finishAllOtherActivity(LoginActivity.class);
                    }
                } else {
                    ToastUtil.showShortToastCenter(MyApplication.getApplication(), errorMessage);
                }
            }
        }
        try {
            onSuccess(t);
        } catch (Exception e2) {
            e2.printStackTrace();
        }
    }

    @Override // io.reactivex.Observer
    public void onSubscribe(Disposable disposable) {
        ProgressDialog progressDialog;
        Context context = this.mContext;
        if ((context instanceof BaseActivity) && ((BaseActivity) context).mCompositeDisposable != null) {
            ((BaseActivity) this.mContext).mCompositeDisposable.add(disposable);
        }
        if (!this.isProgress || (progressDialog = this.progressDialog) == null || progressDialog.isShowing()) {
            return;
        }
        this.progressDialog.show();
    }

    protected abstract void onSuccess(T t);
}
