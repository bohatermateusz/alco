package com.hismith.hismithapp;

import android.bluetooth.BluetoothDevice;
import android.content.BroadcastReceiver;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.text.TextUtils;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.activity.result.ActivityResult;
import androidx.activity.result.ActivityResultCallback;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.media3.extractor.text.ttml.TtmlNode;
import androidx.recyclerview.widget.LinearLayoutManager;
import com.hismith.hismithapp.adapter.BluetoothListAdapter;
import com.hismith.hismithapp.bluetooth.BleManager;
import com.hismith.hismithapp.bluetooth.BleProduct;
import com.hismith.hismithapp.model.BleDeviceModel;
import com.hismith.hismithapp.utils.AppUtil;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.DialogUtil;
import com.hismith.hismithapp.utils.GPSUtil;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.ToastUtil;
import com.hismith.hismithapp.view.SlideRecyclerView;
import com.scwang.smart.refresh.layout.SmartRefreshLayout;
import com.scwang.smart.refresh.layout.api.RefreshLayout;
import com.scwang.smart.refresh.layout.listener.OnRefreshListener;
import com.tbruyelle.rxpermissions2.Permission;
import com.tbruyelle.rxpermissions2.RxPermissions;
import io.reactivex.disposables.Disposable;
import io.reactivex.functions.Consumer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/* loaded from: classes3.dex */
public class BluetootListActivity2 extends BaseActivity implements View.OnClickListener, BleManager.OnScanDeviceListener, BleManager.OnConnectStateListener {
    private static final long SCAN_PERIOD = 10000;
    private static final int WHAT_CONN_DEV = 3;
    public static final int WHAT_START_SCAN = 2;
    public static final int WHAT_STOP_SCAN = 1;
    private BluetoothListAdapter mAdapter;
    private Map<String, String> mBluetoothAddressList;
    private ActivityResultLauncher mBluetoothLauncher;
    private List<BleDeviceModel> mBluetoothList;
    private ImageView mImgBack;
    private LinearLayoutManager mLayoutManager;
    private SlideRecyclerView mRecyclerView;
    private SmartRefreshLayout mRefreshLayout;
    private TextView mTxtName;
    private boolean isScanner = false;
    private Handler mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.BluetootListActivity2.1
        @Override // android.os.Handler.Callback
        public boolean handleMessage(Message message) {
            int i = message.what;
            if (i == 1) {
                BleManager.getInstance(BluetootListActivity2.this).stopLeScan();
                BluetootListActivity2.this.isScanner = false;
                if (BluetootListActivity2.this.mRefreshLayout != null) {
                    BluetootListActivity2.this.mRefreshLayout.finishRefresh();
                }
            } else if (i == 2) {
                BleManager.getInstance(BluetootListActivity2.this).startLeScan();
                BluetootListActivity2.this.isScanner = true;
            } else if (i == 3) {
            }
            return false;
        }
    });
    private BroadcastReceiver mReceiver = new BroadcastReceiver() { // from class: com.hismith.hismithapp.BluetootListActivity2.7
        @Override // android.content.BroadcastReceiver
        public void onReceive(Context context, Intent intent) {
            String action = intent.getAction();
            if (action.equals("android.bluetooth.adapter.action.STATE_CHANGED")) {
                int intExtra = intent.getIntExtra("android.bluetooth.adapter.extra.STATE", 0);
                if (intExtra == 12) {
                    BluetootListActivity2.this.mRefreshLayout.autoRefresh();
                    return;
                } else {
                    if (intExtra == 10 && BluetootListActivity2.this.mHandler.hasMessages(1)) {
                        BluetootListActivity2.this.mHandler.removeMessages(1);
                        BluetootListActivity2.this.mHandler.sendEmptyMessage(1);
                        return;
                    }
                    return;
                }
            }
            if (Constans.RECEIVER_BLE_RECE_DATA.equals(action)) {
                String stringExtra = intent.getStringExtra(Constans.BleType);
                String stringExtra2 = intent.getStringExtra(Constans.BleData0);
                intent.getStringExtra(Constans.BleData1);
                if (("CA".equals(stringExtra) || "CC".equals(stringExtra)) && "00".equals(stringExtra2)) {
                    BluetootListActivity2.this.mAdapter.notifyDataSetChanged();
                }
            }
        }
    };

    /* renamed from: com.hismith.hismithapp.BluetootListActivity2$2, reason: invalid class name */
    class AnonymousClass2 implements BluetoothListAdapter.OnEditNameListener {
        AnonymousClass2() {
        }

        @Override // com.hismith.hismithapp.adapter.BluetoothListAdapter.OnEditNameListener
        public void onEditName(final BluetoothDevice bluetoothDevice, String str, final int i) {
            DialogUtil.showEditDialog2(BluetootListActivity2.this, "changeDeviceName", str, null, null, new DialogUtil.OnEditContentCallback() { // from class: com.hismith.hismithapp.BluetootListActivity2.2.1
                @Override // com.hismith.hismithapp.utils.DialogUtil.OnEditContentCallback
                public void onEditContent(final String str2) {
                    BleProduct connBleProduct;
                    if (BleManager.getInstance(BluetootListActivity2.this).getStatus() != 2 || (connBleProduct = BleManager.getInstance(BluetootListActivity2.this).getConnBleProduct(bluetoothDevice.getAddress())) == null || TextUtils.isEmpty(str2)) {
                        ToastUtil.showShortToastCenter(BluetootListActivity2.this, LanguageUtil.getLanguageByKey(BluetootListActivity2.this, "noChangeDeviceName"));
                        return;
                    }
                    final byte[] bytes = str2.getBytes(StandardCharsets.UTF_8);
                    LogUtil.e("蓝牙名称：" + str2);
                    LogUtil.e("蓝牙名称数组：" + bytes.length + ", " + Arrays.toString(bytes));
                    final byte[] bArr = new byte[18];
                    if (bytes.length > 18) {
                        System.arraycopy(bytes, 0, bArr, 0, 18);
                        LogUtil.e("蓝牙名称18：18, " + Arrays.toString(bArr));
                        connBleProduct.writeName(bArr);
                    } else {
                        connBleProduct.writeName(bytes);
                    }
                    BluetootListActivity2.this.runOnUiThread(new Runnable() { // from class: com.hismith.hismithapp.BluetootListActivity2.2.1.1
                        @Override // java.lang.Runnable
                        public void run() {
                            int length = bytes.length;
                            AnonymousClass1 anonymousClass1 = AnonymousClass1.this;
                            if (length > 18) {
                                ((BleDeviceModel) BluetootListActivity2.this.mBluetoothList.get(i)).setDeviceName(new String(bArr, StandardCharsets.UTF_8));
                            } else {
                                ((BleDeviceModel) BluetootListActivity2.this.mBluetoothList.get(i)).setDeviceName(str2);
                            }
                            BluetootListActivity2.this.mAdapter.notifyDataSetChanged();
                        }
                    });
                }
            });
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public boolean bluetoothIsEnabled() {
        if (BleManager.getInstance(this).getBluetoothAdapter() == null) {
            return false;
        }
        if (BleManager.getInstance(this).getBluetoothAdapter().isEnabled()) {
            return true;
        }
        Intent intent = new Intent("android.bluetooth.adapter.action.REQUEST_ENABLE");
        ActivityResultLauncher activityResultLauncher = this.mBluetoothLauncher;
        if (activityResultLauncher == null) {
            return false;
        }
        activityResultLauncher.launch(intent);
        return false;
    }

    /* JADX INFO: Access modifiers changed from: private */
    public boolean checkBluetoothPermission() {
        PackageManager packageManager = getPackageManager();
        return Build.VERSION.SDK_INT >= 31 ? packageManager.checkPermission("android.permission.BLUETOOTH_SCAN", getPackageName()) == 0 && packageManager.checkPermission("android.permission.BLUETOOTH_CONNECT", getPackageName()) == 0 : packageManager.checkPermission("android.permission.ACCESS_COARSE_LOCATION", getPackageName()) == 0 && packageManager.checkPermission("android.permission.ACCESS_FINE_LOCATION", getPackageName()) == 0;
    }

    private Intent getAppDetailSettingIntent() {
        Intent intent = new Intent();
        intent.addFlags(268435456);
        intent.setAction("android.settings.APPLICATION_DETAILS_SETTINGS");
        intent.setData(Uri.fromParts("package", getPackageName(), null));
        return intent;
    }

    private void gotoHuaweiPermission() {
        try {
            Intent intent = new Intent();
            intent.setFlags(268435456);
            intent.setComponent(new ComponentName("com.huawei.systemmanager", "com.huawei.permissionmanager.ui.MainActivity"));
            startActivity(intent);
        } catch (Exception e) {
            e.printStackTrace();
            startActivity(getAppDetailSettingIntent());
        }
    }

    private void gotoMeizuPermission() {
        Intent intent = new Intent("com.meizu.safe.security.SHOW_APPSEC");
        intent.addCategory("android.intent.category.DEFAULT");
        intent.putExtra("packageName", BuildConfig.APPLICATION_ID);
        try {
            startActivity(intent);
        } catch (Exception e) {
            e.printStackTrace();
            gotoHuaweiPermission();
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void gotoMiuiPermission() {
        Intent intent = new Intent("miui.intent.action.APP_PERM_EDITOR");
        intent.setComponent(new ComponentName("com.miui.securitycenter", "com.miui.permcenter.permissions.AppPermissionsEditorActivity"));
        intent.putExtra("extra_pkgname", getPackageName());
        try {
            startActivity(intent);
        } catch (Exception e) {
            e.printStackTrace();
            gotoMeizuPermission();
        }
    }

    private void init() {
        LinearLayoutManager linearLayoutManager = new LinearLayoutManager(this);
        this.mLayoutManager = linearLayoutManager;
        this.mRecyclerView.setLayoutManager(linearLayoutManager);
        this.mBluetoothList = new ArrayList();
        this.mBluetoothAddressList = new LinkedHashMap();
        BluetoothListAdapter bluetoothListAdapter = new BluetoothListAdapter(this, this.mBluetoothList);
        this.mAdapter = bluetoothListAdapter;
        this.mRecyclerView.setAdapter(bluetoothListAdapter);
        BleManager.getInstance(this).stopLeScan();
        initActivityResult();
        makeFilter();
    }

    private void initActivityResult() {
        this.mBluetoothLauncher = registerForActivityResult(new ActivityResultContracts.StartActivityForResult(), new ActivityResultCallback<ActivityResult>() { // from class: com.hismith.hismithapp.BluetootListActivity2.6
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

    private void initBluetooth() {
        if (BleManager.getInstance(this) == null) {
            finish();
        }
        this.mRefreshLayout.autoRefresh();
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void initBluetoothPermission() {
        Disposable subscribe = new RxPermissions(this).requestEachCombined(Build.VERSION.SDK_INT >= 31 ? new String[]{"android.permission.BLUETOOTH_SCAN", "android.permission.BLUETOOTH_CONNECT"} : new String[]{"android.permission.ACCESS_COARSE_LOCATION", "android.permission.ACCESS_FINE_LOCATION"}).subscribe(new Consumer() { // from class: com.hismith.hismithapp.BluetootListActivity2$$ExternalSyntheticLambda0
            @Override // io.reactivex.functions.Consumer
            public final void accept(Object obj) {
                BluetootListActivity2.this.m554xfd614d9c((Permission) obj);
            }
        });
        if (this.mCompositeDisposable == null || this.mCompositeDisposable.isDisposed()) {
            return;
        }
        this.mCompositeDisposable.add(subscribe);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void initConnDev() {
        for (Map.Entry<String, BleProduct> entry : BleManager.getInstance(this).getBleProducts().entrySet()) {
            BluetoothDevice bluetoothDevice = entry.getValue().getBluetoothDevice();
            if (bluetoothDevice != null && !this.mBluetoothAddressList.containsKey(bluetoothDevice.getAddress())) {
                this.mBluetoothList.add(new BleDeviceModel(entry.getValue().getDeviceName(), bluetoothDevice.getAddress(), entry.getValue().getDeviceCode(), bluetoothDevice));
                this.mBluetoothAddressList.put(bluetoothDevice.getAddress(), entry.getValue().getDeviceName());
                this.mAdapter.notifyDataSetChanged();
            }
        }
    }

    private void initView() {
        this.mImgBack = (ImageView) findViewById(R.id.title_img_back);
        this.mTxtName = (TextView) findViewById(R.id.title_txt_name);
        this.mRefreshLayout = (SmartRefreshLayout) findViewById(R.id.bluetooth_refreshLayout);
        this.mRecyclerView = (SlideRecyclerView) findViewById(R.id.bluetooth_recycler);
        LanguageUtil.setText(this, this.mTxtName, "bluetoothSearch");
        this.mRefreshLayout.setEnableLoadMore(false);
    }

    private void makeFilter() {
        IntentFilter intentFilter = new IntentFilter();
        intentFilter.addAction("android.bluetooth.adapter.action.STATE_CHANGED");
        intentFilter.addAction(Constans.RECEIVER_BLE_RECE_DATA);
        AppUtil.registerReceiver(this, this.mReceiver, intentFilter);
    }

    private void setListener() {
        this.mImgBack.setOnClickListener(this);
        this.mAdapter.setOnEditNameListener(new AnonymousClass2());
        this.mAdapter.setOnItemClickListener(new BluetoothListAdapter.OnItemClickListener() { // from class: com.hismith.hismithapp.BluetootListActivity2.3
            @Override // com.hismith.hismithapp.adapter.BluetoothListAdapter.OnItemClickListener
            public void onItemClick(int i) {
            }
        });
        this.mRefreshLayout.setOnRefreshListener(new OnRefreshListener() { // from class: com.hismith.hismithapp.BluetootListActivity2.4
            @Override // com.scwang.smart.refresh.layout.listener.OnRefreshListener
            public void onRefresh(RefreshLayout refreshLayout) {
                if (!BluetootListActivity2.this.checkBluetoothPermission()) {
                    BluetootListActivity2.this.initBluetoothPermission();
                    BluetootListActivity2.this.mRefreshLayout.finishRefresh(1000);
                    return;
                }
                if (Build.VERSION.SDK_INT < 31 && !GPSUtil.isOpen(BluetootListActivity2.this)) {
                    LogUtil.e("没开启");
                    try {
                        GPSUtil.openGPS(BluetootListActivity2.this);
                        BluetootListActivity2.this.mRefreshLayout.finishRefresh(1000);
                        return;
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
                boolean bluetoothIsEnabled = BluetootListActivity2.this.bluetoothIsEnabled();
                BluetootListActivity2 bluetootListActivity2 = BluetootListActivity2.this;
                if (!bluetoothIsEnabled) {
                    bluetootListActivity2.mRefreshLayout.finishRefresh(1000);
                    return;
                }
                LogUtil.e("搜索 " + BleManager.getInstance(bluetootListActivity2).getOnScanDeviceListener());
                if (BleManager.getInstance(BluetootListActivity2.this).getOnScanDeviceListener() == null || !(BleManager.getInstance(BluetootListActivity2.this).getOnScanDeviceListener() instanceof BluetootListActivity2)) {
                    BleManager.getInstance(BluetootListActivity2.this).setOnScanDeviceListener(BluetootListActivity2.this);
                }
                if (BluetootListActivity2.this.mHandler.hasMessages(1)) {
                    BluetootListActivity2.this.mHandler.removeMessages(1);
                }
                BluetootListActivity2.this.mHandler.sendEmptyMessageDelayed(1, 10000L);
                BluetootListActivity2.this.mHandler.sendEmptyMessage(2);
                BluetootListActivity2.this.mBluetoothList.clear();
                BluetootListActivity2.this.mBluetoothAddressList.clear();
                BluetootListActivity2.this.initConnDev();
                BluetootListActivity2.this.mAdapter.notifyDataSetChanged();
            }
        });
    }

    /* renamed from: lambda$initBluetoothPermission$0$com-hismith-hismithapp-BluetootListActivity2, reason: not valid java name */
    /* synthetic */ void m554xfd614d9c(Permission permission) throws Exception {
        if (permission.granted) {
            return;
        }
        boolean z = permission.shouldShowRequestPermissionRationale;
        DialogUtil.showConfirmDialog(this, "prompt", Build.VERSION.SDK_INT >= 31 ? "nearbyDevicesPermission" : "locationPermissions", null, TtmlNode.TEXT_EMPHASIS_MARK_OPEN, new View.OnClickListener() { // from class: com.hismith.hismithapp.BluetootListActivity2.5
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                BluetootListActivity2.this.gotoMiuiPermission();
                DialogUtil.dismissDialog();
            }
        });
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        if (view.getId() == R.id.title_img_back) {
            this.appManager.finishActivity();
        }
    }

    @Override // com.hismith.hismithapp.bluetooth.BleManager.OnConnectStateListener
    public void onConnected(int i) {
        runOnUiThread(new Runnable() { // from class: com.hismith.hismithapp.BluetootListActivity2.8
            @Override // java.lang.Runnable
            public void run() {
                BluetootListActivity2.this.initConnDev();
                BluetootListActivity2.this.mAdapter.setClicked(false);
                BluetootListActivity2.this.mAdapter.notifyDataSetChanged();
            }
        });
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, androidx.core.app.ComponentActivity, android.app.Activity
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        setContentView(R.layout.activity_bluetoot_list);
        initView();
        init();
        setListener();
        initBluetooth();
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.appcompat.app.AppCompatActivity, androidx.fragment.app.FragmentActivity, android.app.Activity
    protected void onDestroy() {
        super.onDestroy();
        if (this.mHandler.hasMessages(1)) {
            this.mHandler.removeMessages(1);
        }
        BleManager.getInstance(this).startLeScan();
        this.mHandler.removeCallbacksAndMessages(null);
        try {
            unregisterReceiver(this.mReceiver);
        } catch (Exception e) {
            e.printStackTrace();
        }
        BleManager.getInstance(this).setOnScanDeviceListener(null);
    }

    @Override // com.hismith.hismithapp.bluetooth.BleManager.OnConnectStateListener
    public void onDisconnected(int i) {
        runOnUiThread(new Runnable() { // from class: com.hismith.hismithapp.BluetootListActivity2.9
            @Override // java.lang.Runnable
            public void run() {
                BluetootListActivity2.this.mAdapter.setClicked(false);
                BluetootListActivity2.this.mAdapter.notifyDataSetChanged();
            }
        });
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, android.app.Activity
    protected void onPause() {
        super.onPause();
        BleManager.getInstance(this).setOnScanDeviceListener(null);
        BleManager.getInstance(this).setOnConnectStateListener(null);
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, android.app.Activity
    protected void onResume() {
        super.onResume();
        BleManager.getInstance(this).setOnScanDeviceListener(this);
        BleManager.getInstance(this).setOnConnectStateListener(this);
    }

    @Override // com.hismith.hismithapp.bluetooth.BleManager.OnScanDeviceListener
    public void onScanDevice(BluetoothDevice bluetoothDevice, String str, String str2) {
        String address = bluetoothDevice.getAddress();
        if (!this.mBluetoothAddressList.containsKey(address)) {
            this.mBluetoothAddressList.put(address, str);
            this.mBluetoothList.add(new BleDeviceModel(str, bluetoothDevice.getAddress(), str2, bluetoothDevice));
            this.mAdapter.notifyDataSetChanged();
        } else {
            if (TextUtils.isEmpty(str)) {
                return;
            }
            if (TextUtils.isEmpty(this.mBluetoothAddressList.get(address)) || !str.equals(this.mBluetoothAddressList.get(address))) {
                this.mBluetoothAddressList.put(address, str);
                for (int i = 0; i < this.mBluetoothList.size(); i++) {
                    if (this.mBluetoothList.get(i).getDeviceAddress().equals(address)) {
                        this.mBluetoothList.get(i).setDeviceName(str);
                        this.mAdapter.notifyItemChanged(i);
                        return;
                    }
                }
            }
        }
    }
}
