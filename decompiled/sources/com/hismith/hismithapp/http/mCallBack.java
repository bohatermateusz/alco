package com.hismith.hismithapp.http;

import android.app.Activity;
import android.content.Intent;
import com.hismith.hismithapp.BaseActivity;
import com.hismith.hismithapp.LoginActivity;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.ToastUtil;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

/* loaded from: classes3.dex */
public class mCallBack implements Callback<HttpModel> {
    public Activity mActivity;

    public mCallBack(Activity activity) {
        this.mActivity = activity;
    }

    @Override // retrofit2.Callback
    public void onFailure(Call<HttpModel> call, Throwable th) {
        LogUtil.e("请求失败信息: " + th.getMessage());
    }

    @Override // retrofit2.Callback
    public void onResponse(Call<HttpModel> call, Response<HttpModel> response) {
        LogUtil.e("请求成功信息: " + response.body().toString() + "    success:" + response.body().getSuccess());
        if (response.body().getSuccess() != 0 || response.body().getErrors().length <= 0 || response.body().getErrors()[0].getErrorMessage().length() <= 0) {
            return;
        }
        LogUtil.e("请求成功错误信息: " + response.body().getErrors()[0].getErrorCode() + " ,ErrorMessage:" + response.body().getErrors()[0].getErrorMessage());
        ToastUtil.showShortToastCenter(this.mActivity, response.body().getErrors()[0].getErrorMessage());
        if (Constans.errorRelogin.equals(response.body().getErrors()[0].getErrorCode())) {
            this.mActivity.startActivity(new Intent(this.mActivity, (Class<?>) LoginActivity.class));
            ((BaseActivity) this.mActivity).appManager.finishAllOtherActivity(LoginActivity.class);
        }
    }
}
