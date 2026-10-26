package com.hismith.hismithapp.manager;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.text.TextUtils;
import com.facebook.internal.ServerProtocol;
import com.google.android.gms.auth.api.signin.GoogleSignIn;
import com.google.android.gms.auth.api.signin.GoogleSignInAccount;
import com.google.android.gms.auth.api.signin.GoogleSignInOptions;
import com.google.android.gms.common.api.ApiException;
import com.google.android.gms.tasks.OnCompleteListener;
import com.google.android.gms.tasks.Task;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.dialog.ProgressDialog;
import com.hismith.hismithapp.http.HismithApi;
import com.hismith.hismithapp.model.TokenModel;
import com.hismith.hismithapp.utils.Constans;
import com.twitter.sdk.android.core.internal.oauth.OAuthConstants;
import io.reactivex.Observer;
import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.disposables.Disposable;
import io.reactivex.schedulers.Schedulers;
import java.util.HashMap;
import retrofit2.Retrofit;
import retrofit2.adapter.rxjava2.RxJava2CallAdapterFactory;
import retrofit2.converter.gson.GsonConverterFactory;

/* loaded from: classes3.dex */
public class GoogleLoginManager {
    private static final String TAG = "GoogleLoginManager";

    public interface OnGoogleSignOutListener {
        void onSignOutSuccess();
    }

    public interface OnLoginSuccessListener {
        void onSuccessResult(String str);
    }

    public static void GoogleSingOut(Context context, String str, final OnGoogleSignOutListener onGoogleSignOutListener) {
        GoogleSignIn.getClient(context, new GoogleSignInOptions.Builder(GoogleSignInOptions.DEFAULT_SIGN_IN).requestIdToken(str).requestEmail().build()).signOut().addOnCompleteListener((Activity) context, new OnCompleteListener<Void>() { // from class: com.hismith.hismithapp.manager.GoogleLoginManager.2
            @Override // com.google.android.gms.tasks.OnCompleteListener
            public void onComplete(Task<Void> task) {
                OnGoogleSignOutListener.this.onSignOutSuccess();
            }
        });
    }

    public static void getGoogleAccessToken(String str, final OnLoginSuccessListener onLoginSuccessListener) {
        String string = MyApplication.getApplication().getResources().getString(R.string.google_client_id);
        String string2 = MyApplication.getApplication().getResources().getString(R.string.google_client_secret);
        HashMap hashMap = new HashMap();
        hashMap.put("client_id", string);
        hashMap.put(ServerProtocol.DIALOG_PARAM_REDIRECT_URI, "");
        hashMap.put("client_secret", string2);
        hashMap.put(OAuthConstants.PARAM_GRANT_TYPE, "authorization_code");
        hashMap.put("code", str);
        try {
            ((HismithApi) new Retrofit.Builder().baseUrl(Constans.token_url).addCallAdapterFactory(RxJava2CallAdapterFactory.create()).addConverterFactory(GsonConverterFactory.create()).build().create(HismithApi.class)).requestToken(string, string2, str, "", "authorization_code").subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new Observer<TokenModel>() { // from class: com.hismith.hismithapp.manager.GoogleLoginManager.1
                @Override // io.reactivex.Observer
                public void onComplete() {
                }

                @Override // io.reactivex.Observer
                public void onError(Throwable th) {
                }

                @Override // io.reactivex.Observer
                public void onNext(TokenModel tokenModel) {
                    if (tokenModel != null) {
                        String access_token = tokenModel.getAccess_token();
                        if (TextUtils.isEmpty(access_token)) {
                            return;
                        }
                        OnLoginSuccessListener.this.onSuccessResult(access_token);
                    }
                }

                @Override // io.reactivex.Observer
                public void onSubscribe(Disposable disposable) {
                }
            });
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void handleSignInResult(Task<GoogleSignInAccount> task, OnLoginSuccessListener onLoginSuccessListener) {
        try {
            getGoogleAccessToken(task.getResult(ApiException.class).getServerAuthCode(), onLoginSuccessListener);
        } catch (ApiException e) {
            e.printStackTrace();
        }
    }

    public static void initGoogle(Activity activity, String str, int i, ProgressDialog progressDialog) {
        activity.startActivityForResult(GoogleSignIn.getClient(activity, new GoogleSignInOptions.Builder(GoogleSignInOptions.DEFAULT_SIGN_IN).requestServerAuthCode(str).requestIdToken(str).requestEmail().build()).getSignInIntent(), i);
        if (progressDialog == null || progressDialog.isShowing()) {
            return;
        }
        progressDialog.show();
    }

    public static void onActivityResult(int i, Intent intent, int i2, OnLoginSuccessListener onLoginSuccessListener, ProgressDialog progressDialog) {
        if (i == i2) {
            if (progressDialog != null && progressDialog.isShowing()) {
                progressDialog.dismiss();
            }
            handleSignInResult(GoogleSignIn.getSignedInAccountFromIntent(intent), onLoginSuccessListener);
        }
    }
}
