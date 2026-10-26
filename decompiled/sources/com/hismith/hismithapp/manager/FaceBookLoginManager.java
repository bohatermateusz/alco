package com.hismith.hismithapp.manager;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import com.facebook.CallbackManager;
import com.facebook.FacebookCallback;
import com.facebook.FacebookException;
import com.facebook.login.LoginManager;
import com.facebook.login.LoginResult;
import java.util.Arrays;

/* loaded from: classes3.dex */
public class FaceBookLoginManager {
    private static CallbackManager mFaceBookCallBack;
    private static FaceBookLoginManager sInstance;

    public interface OnLoginSuccessListener {
        void onCancel();

        void onError(FacebookException facebookException);

        void onSuccess(LoginResult loginResult);
    }

    private FaceBookLoginManager() {
    }

    public static FaceBookLoginManager getInstance() {
        FaceBookLoginManager faceBookLoginManager;
        FaceBookLoginManager faceBookLoginManager2 = sInstance;
        if (faceBookLoginManager2 != null) {
            return faceBookLoginManager2;
        }
        synchronized (FaceBookLoginManager.class) {
            faceBookLoginManager = sInstance;
            if (faceBookLoginManager == null) {
                faceBookLoginManager = new FaceBookLoginManager();
                sInstance = faceBookLoginManager;
            }
        }
        return faceBookLoginManager;
    }

    public void faceBookLogin(Context context) {
        LoginManager.getInstance().logInWithReadPermissions((Activity) context, Arrays.asList("public_profile"));
    }

    public void initFaceBook(Context context, final OnLoginSuccessListener onLoginSuccessListener) {
        CallbackManager create = CallbackManager.Factory.create();
        mFaceBookCallBack = create;
        if (create != null) {
            LoginManager.getInstance().registerCallback(mFaceBookCallBack, new FacebookCallback<LoginResult>() { // from class: com.hismith.hismithapp.manager.FaceBookLoginManager.1
                @Override // com.facebook.FacebookCallback
                public void onCancel() {
                    OnLoginSuccessListener onLoginSuccessListener2 = onLoginSuccessListener;
                    if (onLoginSuccessListener2 != null) {
                        onLoginSuccessListener2.onCancel();
                    }
                }

                @Override // com.facebook.FacebookCallback
                public void onError(FacebookException facebookException) {
                    OnLoginSuccessListener onLoginSuccessListener2 = onLoginSuccessListener;
                    if (onLoginSuccessListener2 != null) {
                        onLoginSuccessListener2.onError(facebookException);
                    }
                }

                @Override // com.facebook.FacebookCallback
                public void onSuccess(LoginResult loginResult) {
                    OnLoginSuccessListener onLoginSuccessListener2;
                    if (loginResult == null || (onLoginSuccessListener2 = onLoginSuccessListener) == null) {
                        return;
                    }
                    onLoginSuccessListener2.onSuccess(loginResult);
                }
            });
        }
    }

    public void setOnActivityResult(int i, int i2, Intent intent) {
        CallbackManager callbackManager = mFaceBookCallBack;
        if (callbackManager != null) {
            callbackManager.onActivityResult(i, i2, intent);
        }
    }
}
