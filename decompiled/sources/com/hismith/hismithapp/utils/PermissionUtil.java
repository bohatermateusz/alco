package com.hismith.hismithapp.utils;

import android.os.Build;
import androidx.core.content.ContextCompat;
import com.hismith.hismithapp.BaseActivity;
import com.hismith.hismithapp.dialog.TopNotificationDialog;
import com.luck.picture.lib.permissions.PermissionConfig;
import com.tbruyelle.rxpermissions2.Permission;
import com.tbruyelle.rxpermissions2.RxPermissions;
import io.reactivex.disposables.Disposable;
import io.reactivex.functions.Consumer;

/* loaded from: classes3.dex */
public class PermissionUtil {
    private BaseActivity activity;
    private TopNotificationDialog mTopNotificationDialog;

    public interface OnGrantedListener {
        void onGranted();
    }

    public PermissionUtil(BaseActivity baseActivity) {
        this.activity = baseActivity;
    }

    private void showCameraNotificationDialog() {
        TopNotificationDialog topNotificationDialog = new TopNotificationDialog(this.activity, "camera", "cameraPermissionExplanation");
        this.mTopNotificationDialog = topNotificationDialog;
        topNotificationDialog.show();
    }

    private void showRecordNotificationDialog() {
        TopNotificationDialog topNotificationDialog = new TopNotificationDialog(this.activity, "mic", "micPermissionExplanation");
        this.mTopNotificationDialog = topNotificationDialog;
        topNotificationDialog.show();
    }

    private void showStorageNotificationDialog() {
        TopNotificationDialog topNotificationDialog = new TopNotificationDialog(this.activity, "storage", "storagePermissionExplanation");
        this.mTopNotificationDialog = topNotificationDialog;
        topNotificationDialog.show();
    }

    public void initCameraPermission(final OnGrantedListener onGrantedListener) {
        Disposable subscribe = new RxPermissions(this.activity).requestEachCombined("android.permission.CAMERA").subscribe(new Consumer() { // from class: com.hismith.hismithapp.utils.PermissionUtil$$ExternalSyntheticLambda1
            @Override // io.reactivex.functions.Consumer
            public final void accept(Object obj) {
                PermissionUtil.this.m622xb6011bee(onGrantedListener, (Permission) obj);
            }
        });
        if (this.activity.mCompositeDisposable == null || this.activity.mCompositeDisposable.isDisposed()) {
            return;
        }
        this.activity.mCompositeDisposable.add(subscribe);
    }

    public void initImagePermission(final OnGrantedListener onGrantedListener) {
        RxPermissions rxPermissions = new RxPermissions(this.activity);
        if (Build.VERSION.SDK_INT >= 33) {
            if (onGrantedListener != null) {
                onGrantedListener.onGranted();
            }
        } else {
            Disposable subscribe = rxPermissions.requestEachCombined(PermissionConfig.READ_EXTERNAL_STORAGE).subscribe(new Consumer() { // from class: com.hismith.hismithapp.utils.PermissionUtil$$ExternalSyntheticLambda0
                @Override // io.reactivex.functions.Consumer
                public final void accept(Object obj) {
                    PermissionUtil.this.m623xde4c5da9(onGrantedListener, (Permission) obj);
                }
            });
            if (this.activity.mCompositeDisposable == null || this.activity.mCompositeDisposable.isDisposed()) {
                return;
            }
            this.activity.mCompositeDisposable.add(subscribe);
        }
    }

    public void initRecordPermission(final OnGrantedListener onGrantedListener) {
        if (ContextCompat.checkSelfPermission(this.activity, "android.permission.RECORD_AUDIO") == 0) {
            if (onGrantedListener != null) {
                onGrantedListener.onGranted();
            }
        } else {
            Disposable subscribe = new RxPermissions(this.activity).requestEachCombined("android.permission.RECORD_AUDIO").subscribe(new Consumer() { // from class: com.hismith.hismithapp.utils.PermissionUtil$$ExternalSyntheticLambda2
                @Override // io.reactivex.functions.Consumer
                public final void accept(Object obj) {
                    PermissionUtil.this.m624xc5d1139b(onGrantedListener, (Permission) obj);
                }
            });
            if (this.activity.mCompositeDisposable == null || this.activity.mCompositeDisposable.isDisposed()) {
                return;
            }
            this.activity.mCompositeDisposable.add(subscribe);
        }
    }

    /* renamed from: lambda$initCameraPermission$1$com-hismith-hismithapp-utils-PermissionUtil, reason: not valid java name */
    /* synthetic */ void m622xb6011bee(OnGrantedListener onGrantedListener, Permission permission) throws Exception {
        if (permission.granted) {
            LogUtil.e("获取权限成功");
            if (onGrantedListener != null) {
                onGrantedListener.onGranted();
            }
        } else if (permission.shouldShowRequestPermissionRationale) {
            LogUtil.e("获取权限失败，但是用户没有勾选“不再询问”");
            LogUtil2.e("获取权限失败，但是用户没有勾选“不再询问”");
        } else {
            ToastUtil.showShortToast(this.activity, LanguageUtil.getLanguageByKey(this.activity, "cameraPermission"));
            LogUtil.e("权限申请失败，用户勾选了“不再询问”");
            LogUtil2.e("权限申请失败，用户勾选了“不再询问”");
        }
        TopNotificationDialog topNotificationDialog = this.mTopNotificationDialog;
        if (topNotificationDialog == null || !topNotificationDialog.isShowing()) {
            return;
        }
        this.mTopNotificationDialog.dismiss();
    }

    /* renamed from: lambda$initImagePermission$0$com-hismith-hismithapp-utils-PermissionUtil, reason: not valid java name */
    /* synthetic */ void m623xde4c5da9(OnGrantedListener onGrantedListener, Permission permission) throws Exception {
        if (permission.granted) {
            LogUtil.e("获取权限成功");
            if (onGrantedListener != null) {
                onGrantedListener.onGranted();
            }
        } else if (permission.shouldShowRequestPermissionRationale) {
            LogUtil.e("获取权限失败，但是用户没有勾选“不再询问”");
        } else {
            ToastUtil.showShortToast(this.activity, LanguageUtil.getLanguageByKey(this.activity, "storagePermissionDenied"));
            LogUtil.e("权限申请失败，用户勾选了“不再询问”");
        }
        TopNotificationDialog topNotificationDialog = this.mTopNotificationDialog;
        if (topNotificationDialog == null || !topNotificationDialog.isShowing()) {
            return;
        }
        this.mTopNotificationDialog.dismiss();
    }

    /* renamed from: lambda$initRecordPermission$2$com-hismith-hismithapp-utils-PermissionUtil, reason: not valid java name */
    /* synthetic */ void m624xc5d1139b(OnGrantedListener onGrantedListener, Permission permission) throws Exception {
        if (permission.granted) {
            LogUtil.e("获取权限成功");
            if (onGrantedListener != null) {
                onGrantedListener.onGranted();
            }
        } else if (permission.shouldShowRequestPermissionRationale) {
            LogUtil.e("获取权限失败，但是用户没有勾选“不再询问”");
        } else {
            ToastUtil.showShortToast(this.activity, LanguageUtil.getLanguageByKey(this.activity, "microPermission"));
            LogUtil.e("权限申请失败，用户勾选了“不再询问”");
        }
        TopNotificationDialog topNotificationDialog = this.mTopNotificationDialog;
        if (topNotificationDialog == null || !topNotificationDialog.isShowing()) {
            return;
        }
        this.mTopNotificationDialog.dismiss();
    }
}
