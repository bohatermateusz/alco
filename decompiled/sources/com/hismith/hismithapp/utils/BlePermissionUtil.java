package com.hismith.hismithapp.utils;

import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.Resources;
import android.net.Uri;
import android.os.Build;
import android.view.View;
import androidx.activity.result.ActivityResult;
import androidx.activity.result.ActivityResultCallback;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.constraintlayout.core.motion.utils.TypedValues;
import androidx.media3.extractor.text.ttml.TtmlNode;
import com.hismith.hismithapp.BaseActivity;
import com.hismith.hismithapp.BuildConfig;
import com.hismith.hismithapp.bluetooth.BleManager;
import com.tbruyelle.rxpermissions2.Permission;
import com.tbruyelle.rxpermissions2.RxPermissions;
import io.reactivex.disposables.Disposable;
import io.reactivex.functions.Consumer;

/* loaded from: classes3.dex */
public class BlePermissionUtil {
    private BaseActivity activity;
    private ActivityResultLauncher mBluetoothLauncher;

    public interface OnPermissionListener {
        void onGranted();
    }

    public BlePermissionUtil(BaseActivity baseActivity) {
        this.activity = baseActivity;
        initActivityResult();
    }

    private void enableBluetooth() {
        Intent intent = new Intent("android.bluetooth.adapter.action.REQUEST_ENABLE");
        ActivityResultLauncher activityResultLauncher = this.mBluetoothLauncher;
        if (activityResultLauncher != null) {
            activityResultLauncher.launch(intent);
        }
    }

    private Intent getAppDetailSettingIntent(Context context) {
        Intent intent = new Intent();
        intent.addFlags(268435456);
        intent.setAction("android.settings.APPLICATION_DETAILS_SETTINGS");
        intent.setData(Uri.fromParts("package", context.getPackageName(), null));
        return intent;
    }

    private void gotoHuaweiPermission(Context context) {
        try {
            Intent intent = new Intent();
            intent.setFlags(268435456);
            intent.setComponent(new ComponentName("com.huawei.systemmanager", "com.huawei.permissionmanager.ui.MainActivity"));
            context.startActivity(intent);
        } catch (Exception e) {
            e.printStackTrace();
            context.startActivity(getAppDetailSettingIntent(context));
        }
    }

    private void gotoMeizuPermission(Context context) {
        Intent intent = new Intent("com.meizu.safe.security.SHOW_APPSEC");
        intent.addCategory("android.intent.category.DEFAULT");
        intent.putExtra("packageName", BuildConfig.APPLICATION_ID);
        try {
            context.startActivity(intent);
        } catch (Exception e) {
            e.printStackTrace();
            gotoHuaweiPermission(context);
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void gotoMiuiPermission(Context context) {
        Intent intent = new Intent("miui.intent.action.APP_PERM_EDITOR");
        intent.setComponent(new ComponentName("com.miui.securitycenter", "com.miui.permcenter.permissions.AppPermissionsEditorActivity"));
        intent.putExtra("extra_pkgname", context.getPackageName());
        try {
            context.startActivity(intent);
        } catch (Exception e) {
            e.printStackTrace();
            gotoMeizuPermission(context);
        }
    }

    private void initActivityResult() {
        this.mBluetoothLauncher = this.activity.registerForActivityResult(new ActivityResultContracts.StartActivityForResult(), new ActivityResultCallback<ActivityResult>() { // from class: com.hismith.hismithapp.utils.BlePermissionUtil.1
            @Override // androidx.activity.result.ActivityResultCallback
            public void onActivityResult(ActivityResult activityResult) {
                if (activityResult.getResultCode() == -1) {
                    LogUtil.e("蓝牙打开");
                } else if (activityResult.getResultCode() == 0) {
                    LogUtil.e("蓝牙打不开");
                }
            }
        });
    }

    public static boolean isHarmony(Context context) {
        try {
            return context.getString(Resources.getSystem().getIdentifier("config_os_brand", TypedValues.Custom.S_STRING, "android")).equals("harmony");
        } catch (Exception unused) {
            return false;
        }
    }

    public void enableBluetoothPermission(final OnPermissionListener onPermissionListener) {
        Disposable subscribe = new RxPermissions(this.activity).requestEachCombined(Build.VERSION.SDK_INT >= 31 ? new String[]{"android.permission.BLUETOOTH_SCAN", "android.permission.BLUETOOTH_CONNECT"} : new String[]{"android.permission.ACCESS_COARSE_LOCATION", "android.permission.ACCESS_FINE_LOCATION"}).subscribe(new Consumer() { // from class: com.hismith.hismithapp.utils.BlePermissionUtil$$ExternalSyntheticLambda0
            @Override // io.reactivex.functions.Consumer
            public final void accept(Object obj) {
                BlePermissionUtil.this.m621x6b2d6e74(onPermissionListener, (Permission) obj);
            }
        });
        if (this.activity.mCompositeDisposable == null || this.activity.mCompositeDisposable.isDisposed()) {
            return;
        }
        this.activity.mCompositeDisposable.add(subscribe);
    }

    public boolean isBluetoothPermission() {
        PackageManager packageManager = this.activity.getPackageManager();
        return packageManager.checkPermission("android.permission.BLUETOOTH_SCAN", this.activity.getPackageName()) == 0 && packageManager.checkPermission("android.permission.BLUETOOTH_CONNECT", this.activity.getPackageName()) == 0;
    }

    /* renamed from: lambda$enableBluetoothPermission$0$com-hismith-hismithapp-utils-BlePermissionUtil, reason: not valid java name */
    /* synthetic */ void m621x6b2d6e74(OnPermissionListener onPermissionListener, Permission permission) throws Exception {
        if (!permission.granted) {
            if (permission.shouldShowRequestPermissionRationale) {
                return;
            }
            DialogUtil.showConfirmDialog(this.activity, "prompt", Build.VERSION.SDK_INT >= 31 ? "nearbyDevicesPermission" : "locationPermissions", null, TtmlNode.TEXT_EMPHASIS_MARK_OPEN, new View.OnClickListener() { // from class: com.hismith.hismithapp.utils.BlePermissionUtil.2
                @Override // android.view.View.OnClickListener
                public void onClick(View view) {
                    BlePermissionUtil blePermissionUtil = BlePermissionUtil.this;
                    blePermissionUtil.gotoMiuiPermission(blePermissionUtil.activity);
                    DialogUtil.dismissDialog();
                }
            });
            return;
        }
        if (BleManager.getInstance(this.activity).getBluetoothAdapter() != null && !BleManager.getInstance(this.activity).getBluetoothAdapter().isEnabled()) {
            enableBluetooth();
        }
        if (onPermissionListener != null) {
            onPermissionListener.onGranted();
        }
    }
}
