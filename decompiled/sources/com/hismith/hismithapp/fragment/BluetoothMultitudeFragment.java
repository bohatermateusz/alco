package com.hismith.hismithapp.fragment;

import android.app.Activity;
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
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.activity.result.ActivityResult;
import androidx.activity.result.ActivityResultCallback;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.media3.extractor.text.ttml.TtmlNode;
import androidx.recyclerview.widget.LinearLayoutManager;
import com.facebook.appevents.integrity.IntegrityManager;
import com.hismith.hismithapp.BuildConfig;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.adapter.BluetoothMultitudeAdapter;
import com.hismith.hismithapp.bluetooth.BleManager;
import com.hismith.hismithapp.bluetooth.BleProduct;
import com.hismith.hismithapp.device_rule.SpeedChannel;
import com.hismith.hismithapp.dialog.SelectMotorDialog;
import com.hismith.hismithapp.fragment.BluetoothMultitudeFragment;
import com.hismith.hismithapp.manager.ControlRuleManager;
import com.hismith.hismithapp.manager.ProductManager;
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
import io.reactivex.disposables.CompositeDisposable;
import io.reactivex.disposables.Disposable;
import io.reactivex.functions.Consumer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/* loaded from: classes3.dex */
public class BluetoothMultitudeFragment extends LazyLoadFragment implements View.OnClickListener, BleManager.OnScanDeviceListener {
    private static final long SCAN_PERIOD = 10000;
    public static final int WHAT_START_SCAN = 2;
    public static final int WHAT_STOP_SCAN = 1;
    private FragmentInteractionListener interactionListener;
    private Activity mActivity;
    private BluetoothMultitudeAdapter mAdapter;
    private Map<String, String> mBluetoothAddressList;
    private ActivityResultLauncher mBluetoothLauncher;
    private List<BleDeviceModel> mBluetoothList;
    public CompositeDisposable mCompositeDisposable;
    private Context mContext;
    private LinearLayoutManager mLayoutManager;
    private SmartRefreshLayout mRefreshLayout;
    private SlideRecyclerView recycler_bluetooth;
    private ImageView title_img_back;
    private TextView title_txt_name;
    private Handler mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.fragment.BluetoothMultitudeFragment.1
        @Override // android.os.Handler.Callback
        public boolean handleMessage(Message message) {
            int i = message.what;
            if (i != 1) {
                if (i != 2) {
                    return false;
                }
                BleManager.getInstance(BluetoothMultitudeFragment.this.mContext).startLeScan();
                return false;
            }
            BleManager.getInstance(BluetoothMultitudeFragment.this.mContext).stopLeScan();
            if (BluetoothMultitudeFragment.this.mRefreshLayout == null) {
                return false;
            }
            BluetoothMultitudeFragment.this.mRefreshLayout.finishRefresh();
            return false;
        }
    });
    private BroadcastReceiver mReceiver = new BroadcastReceiver() { // from class: com.hismith.hismithapp.fragment.BluetoothMultitudeFragment.8
        @Override // android.content.BroadcastReceiver
        public void onReceive(Context context, Intent intent) {
            String action = intent.getAction();
            if (action.equals("android.bluetooth.adapter.action.STATE_CHANGED")) {
                int intExtra = intent.getIntExtra("android.bluetooth.adapter.extra.STATE", 0);
                if (intExtra == 12) {
                    BluetoothMultitudeFragment.this.mRefreshLayout.autoRefresh();
                    return;
                } else {
                    if (intExtra == 10 && BluetoothMultitudeFragment.this.mHandler.hasMessages(1)) {
                        BluetoothMultitudeFragment.this.mHandler.removeMessages(1);
                        BluetoothMultitudeFragment.this.mHandler.sendEmptyMessage(1);
                        return;
                    }
                    return;
                }
            }
            if (Constans.RECEIVER_BLE_RECE_DATA.equals(action)) {
                LogUtil.e("广播RECEIVER_BLE_RECE_DATA BluetoothMultitudefragment");
                String stringExtra = intent.getStringExtra(Constans.BleType);
                String stringExtra2 = intent.getStringExtra(Constans.BleData0);
                intent.getStringExtra(Constans.BleData1);
                String stringExtra3 = intent.getStringExtra(IntegrityManager.INTEGRITY_TYPE_ADDRESS);
                for (int i = 0; BluetoothMultitudeFragment.this.mBluetoothList != null && i < BluetoothMultitudeFragment.this.mBluetoothList.size(); i++) {
                    if (((BleDeviceModel) BluetoothMultitudeFragment.this.mBluetoothList.get(i)).getDeviceAddress().equals(stringExtra3) && (("CA".equals(stringExtra) || "CC".equals(stringExtra)) && "00".equals(stringExtra2) && BluetoothMultitudeFragment.this.mAdapter != null)) {
                        BluetoothMultitudeFragment.this.mAdapter.notifyItemChanged(i, "battery");
                    }
                }
            }
        }
    };

    /* renamed from: com.hismith.hismithapp.fragment.BluetoothMultitudeFragment$3, reason: invalid class name */
    class AnonymousClass3 implements BluetoothMultitudeAdapter.MotorClickListener {
        AnonymousClass3() {
        }

        /* renamed from: lambda$onMotor1Click$0$com-hismith-hismithapp-fragment-BluetoothMultitudeFragment$3, reason: not valid java name */
        /* synthetic */ void m595xcb31373a(int i, String str, ControlRuleManager.ControlRule controlRule) {
            if (BluetoothMultitudeFragment.this.mAdapter != null) {
                BluetoothMultitudeFragment.this.mAdapter.notifyItemChanged(i);
            }
        }

        /* renamed from: lambda$onMotor2Click$1$com-hismith-hismithapp-fragment-BluetoothMultitudeFragment$3, reason: not valid java name */
        /* synthetic */ void m596x50c1365a(int i, String str, ControlRuleManager.ControlRule controlRule) {
            if (BluetoothMultitudeFragment.this.mAdapter != null) {
                BluetoothMultitudeFragment.this.mAdapter.notifyItemChanged(i);
            }
        }

        @Override // com.hismith.hismithapp.adapter.BluetoothMultitudeAdapter.MotorClickListener
        public void onMotor1Click(int i, String str) {
            try {
                SelectMotorDialog selectMotorDialog = new SelectMotorDialog(BluetoothMultitudeFragment.this.mContext, i, str, ControlRuleManager.getInstance().getDeviceRule(str).m606clone(), SpeedChannel.CHANNEL_A, "functionSet", null, null);
                selectMotorDialog.setOnRuleChangedListener(new SelectMotorDialog.OnRuleChangedListener() { // from class: com.hismith.hismithapp.fragment.BluetoothMultitudeFragment$3$$ExternalSyntheticLambda0
                    @Override // com.hismith.hismithapp.dialog.SelectMotorDialog.OnRuleChangedListener
                    public final void onConfirm(int i2, String str2, ControlRuleManager.ControlRule controlRule) {
                        BluetoothMultitudeFragment.AnonymousClass3.this.m595xcb31373a(i2, str2, controlRule);
                    }
                });
                selectMotorDialog.show();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        @Override // com.hismith.hismithapp.adapter.BluetoothMultitudeAdapter.MotorClickListener
        public void onMotor2Click(int i, String str) {
            try {
                SelectMotorDialog selectMotorDialog = new SelectMotorDialog(BluetoothMultitudeFragment.this.mContext, i, str, ControlRuleManager.getInstance().getDeviceRule(str).m606clone(), SpeedChannel.CHANNEL_B, "functionSet", null, null);
                selectMotorDialog.setOnRuleChangedListener(new SelectMotorDialog.OnRuleChangedListener() { // from class: com.hismith.hismithapp.fragment.BluetoothMultitudeFragment$3$$ExternalSyntheticLambda1
                    @Override // com.hismith.hismithapp.dialog.SelectMotorDialog.OnRuleChangedListener
                    public final void onConfirm(int i2, String str2, ControlRuleManager.ControlRule controlRule) {
                        BluetoothMultitudeFragment.AnonymousClass3.this.m596x50c1365a(i2, str2, controlRule);
                    }
                });
                selectMotorDialog.show();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    /* renamed from: com.hismith.hismithapp.fragment.BluetoothMultitudeFragment$4, reason: invalid class name */
    class AnonymousClass4 implements BluetoothMultitudeAdapter.OnEditNameListener {
        AnonymousClass4() {
        }

        @Override // com.hismith.hismithapp.adapter.BluetoothMultitudeAdapter.OnEditNameListener
        public void onEditName(final BluetoothDevice bluetoothDevice, String str, final int i) {
            DialogUtil.showEditDialog2(BluetoothMultitudeFragment.this.mActivity, "changeDeviceName", str, null, null, new DialogUtil.OnEditContentCallback() { // from class: com.hismith.hismithapp.fragment.BluetoothMultitudeFragment.4.1
                @Override // com.hismith.hismithapp.utils.DialogUtil.OnEditContentCallback
                public void onEditContent(final String str2) {
                    final BleProduct connBleProduct;
                    if (BleManager.getInstance(BluetoothMultitudeFragment.this.mContext).getStatus() != 2 || (connBleProduct = BleManager.getInstance(BluetoothMultitudeFragment.this.mContext).getConnBleProduct(bluetoothDevice.getAddress())) == null || TextUtils.isEmpty(str2)) {
                        ToastUtil.showShortToastCenter(BluetoothMultitudeFragment.this.mContext, LanguageUtil.getLanguageByKey(BluetoothMultitudeFragment.this.mContext, "noChangeDeviceName"));
                        return;
                    }
                    final byte[] bytes = str2.getBytes(StandardCharsets.UTF_8);
                    LogUtil.e("蓝牙名称：" + str2);
                    LogUtil.e("蓝牙名称数组：" + bytes.length + ", " + Arrays.toString(bytes));
                    final byte[] bArr = new byte[16];
                    if (bytes.length > 16) {
                        System.arraycopy(bytes, 0, bArr, 0, 16);
                        LogUtil.e("蓝牙名称16：16, " + Arrays.toString(bArr));
                        connBleProduct.writeName(bArr);
                    } else {
                        connBleProduct.writeName(bytes);
                    }
                    BluetoothMultitudeFragment.this.mActivity.runOnUiThread(new Runnable() { // from class: com.hismith.hismithapp.fragment.BluetoothMultitudeFragment.4.1.1
                        @Override // java.lang.Runnable
                        public void run() {
                            if (bytes.length > 16) {
                                String str3 = new String(bArr, StandardCharsets.UTF_8);
                                ((BleDeviceModel) BluetoothMultitudeFragment.this.mBluetoothList.get(i)).setDeviceName(str3);
                                connBleProduct.setmDeviceName(str3);
                            } else {
                                ((BleDeviceModel) BluetoothMultitudeFragment.this.mBluetoothList.get(i)).setDeviceName(str2);
                                connBleProduct.setmDeviceName(str2);
                            }
                            BluetoothMultitudeFragment.this.mAdapter.notifyDataSetChanged();
                        }
                    });
                }
            }, 16);
        }
    }

    public interface FragmentInteractionListener {
        void onFragmentRemoved(String str);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public boolean bluetoothIsEnabled() {
        if (BleManager.getInstance(this.mContext).getBluetoothAdapter() == null) {
            return false;
        }
        if (BleManager.getInstance(this.mContext).getBluetoothAdapter().isEnabled()) {
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
        PackageManager packageManager = this.mContext.getPackageManager();
        int i = Build.VERSION.SDK_INT;
        Context context = this.mContext;
        return i >= 31 ? packageManager.checkPermission("android.permission.BLUETOOTH_SCAN", context.getPackageName()) == 0 && packageManager.checkPermission("android.permission.BLUETOOTH_CONNECT", this.mContext.getPackageName()) == 0 : packageManager.checkPermission("android.permission.ACCESS_COARSE_LOCATION", context.getPackageName()) == 0 && packageManager.checkPermission("android.permission.ACCESS_FINE_LOCATION", this.mContext.getPackageName()) == 0;
    }

    private Intent getAppDetailSettingIntent() {
        Intent intent = new Intent();
        intent.addFlags(268435456);
        intent.setAction("android.settings.APPLICATION_DETAILS_SETTINGS");
        intent.setData(Uri.fromParts("package", this.mContext.getPackageName(), null));
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
        intent.putExtra("extra_pkgname", this.mContext.getPackageName());
        try {
            startActivity(intent);
        } catch (Exception e) {
            e.printStackTrace();
            gotoMeizuPermission();
        }
    }

    private void init() {
        this.mCompositeDisposable = new CompositeDisposable();
        LinearLayoutManager linearLayoutManager = new LinearLayoutManager(this.mContext);
        this.mLayoutManager = linearLayoutManager;
        this.recycler_bluetooth.setLayoutManager(linearLayoutManager);
        this.mBluetoothList = new ArrayList();
        this.mBluetoothAddressList = new LinkedHashMap();
        BluetoothMultitudeAdapter bluetoothMultitudeAdapter = new BluetoothMultitudeAdapter(this.mContext, this.mBluetoothList);
        this.mAdapter = bluetoothMultitudeAdapter;
        this.recycler_bluetooth.setAdapter(bluetoothMultitudeAdapter);
        LanguageUtil.setText(this.mContext, this.title_txt_name, "bluetoothSearch");
        this.mRefreshLayout.setEnableLoadMore(false);
        makeFilter();
        refreshData();
    }

    private void initActivityResult() {
        this.mBluetoothLauncher = registerForActivityResult(new ActivityResultContracts.StartActivityForResult(), new ActivityResultCallback<ActivityResult>() { // from class: com.hismith.hismithapp.fragment.BluetoothMultitudeFragment.7
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

    /* JADX INFO: Access modifiers changed from: private */
    public void initBluetoothPermission() {
        Disposable subscribe = new RxPermissions(this).requestEachCombined(Build.VERSION.SDK_INT >= 31 ? new String[]{"android.permission.BLUETOOTH_SCAN", "android.permission.BLUETOOTH_CONNECT"} : new String[]{"android.permission.ACCESS_COARSE_LOCATION", "android.permission.ACCESS_FINE_LOCATION"}).subscribe(new Consumer() { // from class: com.hismith.hismithapp.fragment.BluetoothMultitudeFragment$$ExternalSyntheticLambda0
            @Override // io.reactivex.functions.Consumer
            public final void accept(Object obj) {
                BluetoothMultitudeFragment.this.m594xbf20966b((Permission) obj);
            }
        });
        CompositeDisposable compositeDisposable = this.mCompositeDisposable;
        if (compositeDisposable == null || compositeDisposable.isDisposed()) {
            return;
        }
        this.mCompositeDisposable.add(subscribe);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void initConnDev() {
        this.mBluetoothList.clear();
        this.mBluetoothAddressList.clear();
        for (Map.Entry<String, BleProduct> entry : BleManager.getInstance(this.mContext).getBleProducts().entrySet()) {
            BluetoothDevice bluetoothDevice = entry.getValue().getBluetoothDevice();
            if (bluetoothDevice != null && !this.mBluetoothAddressList.containsKey(bluetoothDevice.getAddress())) {
                this.mBluetoothList.add(new BleDeviceModel(entry.getValue().getDeviceName(), bluetoothDevice.getAddress(), entry.getValue().getDeviceCode(), bluetoothDevice, entry.getValue().getmProductModel()));
                this.mBluetoothAddressList.put(bluetoothDevice.getAddress(), entry.getValue().getDeviceName());
            }
        }
        BluetoothMultitudeAdapter bluetoothMultitudeAdapter = this.mAdapter;
        if (bluetoothMultitudeAdapter != null) {
            bluetoothMultitudeAdapter.notifyDataSetChanged();
        }
    }

    private void initView(View view) {
        this.title_img_back = (ImageView) view.findViewById(R.id.title_img_back);
        this.title_txt_name = (TextView) view.findViewById(R.id.title_txt_name);
        this.mRefreshLayout = (SmartRefreshLayout) view.findViewById(R.id.refresh_layout);
        this.recycler_bluetooth = (SlideRecyclerView) view.findViewById(R.id.recycler_bluetooth);
    }

    private void makeFilter() {
        IntentFilter intentFilter = new IntentFilter();
        intentFilter.addAction("android.bluetooth.adapter.action.STATE_CHANGED");
        intentFilter.addAction(Constans.RECEIVER_BLE_RECE_DATA);
        AppUtil.registerReceiver(this.mContext, this.mReceiver, intentFilter);
    }

    private void refreshData() {
        this.mRefreshLayout.autoRefresh();
    }

    private void removeSelf() {
        if (getParentFragmentManager() == null || !isAdded()) {
            return;
        }
        getParentFragmentManager().beginTransaction().remove(this).commit();
        FragmentInteractionListener fragmentInteractionListener = this.interactionListener;
        if (fragmentInteractionListener != null) {
            fragmentInteractionListener.onFragmentRemoved(getTag());
        }
    }

    private void setListener() {
        this.title_img_back.setOnClickListener(this);
        this.mRefreshLayout.setOnRefreshListener(new OnRefreshListener() { // from class: com.hismith.hismithapp.fragment.BluetoothMultitudeFragment.2
            @Override // com.scwang.smart.refresh.layout.listener.OnRefreshListener
            public void onRefresh(RefreshLayout refreshLayout) {
                if (!BluetoothMultitudeFragment.this.checkBluetoothPermission()) {
                    BluetoothMultitudeFragment.this.initBluetoothPermission();
                    BluetoothMultitudeFragment.this.mRefreshLayout.finishRefresh(1000);
                    return;
                }
                if (Build.VERSION.SDK_INT < 31 && !GPSUtil.isOpen(BluetoothMultitudeFragment.this.mContext)) {
                    LogUtil.e("没开启");
                    try {
                        GPSUtil.openGPS(BluetoothMultitudeFragment.this.mContext);
                        BluetoothMultitudeFragment.this.mRefreshLayout.finishRefresh(1000);
                        return;
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
                boolean bluetoothIsEnabled = BluetoothMultitudeFragment.this.bluetoothIsEnabled();
                BluetoothMultitudeFragment bluetoothMultitudeFragment = BluetoothMultitudeFragment.this;
                if (!bluetoothIsEnabled) {
                    bluetoothMultitudeFragment.mRefreshLayout.finishRefresh(1000);
                    return;
                }
                bluetoothMultitudeFragment.initConnDev();
                LogUtil.e("搜索 " + BleManager.getInstance(BluetoothMultitudeFragment.this.mContext).getOnScanDeviceListener());
                if (BleManager.getInstance(BluetoothMultitudeFragment.this.mContext).getOnScanDeviceListener() == null || !(BleManager.getInstance(BluetoothMultitudeFragment.this.mContext).getOnScanDeviceListener() instanceof BluetoothMultitudeFragment)) {
                    BleManager.getInstance(BluetoothMultitudeFragment.this.mContext).setOnScanDeviceListener(BluetoothMultitudeFragment.this);
                }
                if (BluetoothMultitudeFragment.this.mHandler.hasMessages(1)) {
                    BluetoothMultitudeFragment.this.mHandler.removeMessages(1);
                }
                BluetoothMultitudeFragment.this.mHandler.sendEmptyMessageDelayed(1, 10000L);
                BluetoothMultitudeFragment.this.mHandler.sendEmptyMessage(2);
            }
        });
        this.mAdapter.setMotorClickListener(new AnonymousClass3());
        this.mAdapter.setOnEditNameListener(new AnonymousClass4());
    }

    /* renamed from: lambda$initBluetoothPermission$0$com-hismith-hismithapp-fragment-BluetoothMultitudeFragment, reason: not valid java name */
    /* synthetic */ void m594xbf20966b(Permission permission) throws Exception {
        if (permission.granted) {
            return;
        }
        boolean z = permission.shouldShowRequestPermissionRationale;
        DialogUtil.showConfirmDialog(this.mContext, "prompt", Build.VERSION.SDK_INT >= 31 ? "nearbyDevicesPermission" : "locationPermissions", null, TtmlNode.TEXT_EMPHASIS_MARK_OPEN, new View.OnClickListener() { // from class: com.hismith.hismithapp.fragment.BluetoothMultitudeFragment.6
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                BluetoothMultitudeFragment.this.gotoMiuiPermission();
                DialogUtil.dismissDialog();
            }
        });
    }

    @Override // com.hismith.hismithapp.fragment.LazyLoadFragment
    protected void lazyLoad() {
        init();
        setListener();
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        if (R.id.title_img_back == view.getId()) {
            removeSelf();
        }
    }

    public void onConnected(int i) {
        LogUtil.e("多连接 onConnected " + i);
        sortBluetoothList();
        BluetoothMultitudeAdapter bluetoothMultitudeAdapter = this.mAdapter;
        if (bluetoothMultitudeAdapter != null) {
            bluetoothMultitudeAdapter.setClicked(false);
            this.mAdapter.notifyDataSetChanged();
        }
    }

    @Override // com.hismith.hismithapp.fragment.LazyLoadFragment, androidx.fragment.app.Fragment
    public void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        this.mActivity = getActivity();
        this.mContext = getContext();
        if (getActivity() instanceof FragmentInteractionListener) {
            this.interactionListener = (FragmentInteractionListener) getActivity();
        }
        initActivityResult();
    }

    @Override // androidx.fragment.app.Fragment
    public View onCreateView(LayoutInflater layoutInflater, ViewGroup viewGroup, Bundle bundle) {
        View inflate = layoutInflater.inflate(R.layout.fragment_bluetooth_multitude, viewGroup, false);
        initView(inflate);
        return inflate;
    }

    @Override // androidx.fragment.app.Fragment
    public void onDestroy() {
        super.onDestroy();
        CompositeDisposable compositeDisposable = this.mCompositeDisposable;
        if (compositeDisposable != null) {
            compositeDisposable.clear();
            this.mCompositeDisposable = null;
        }
        this.interactionListener = null;
        BleManager.getInstance(this.mContext).setOnScanDeviceListener(null);
        if (this.mHandler.hasMessages(1)) {
            this.mHandler.removeMessages(1);
        }
        BleManager.getInstance(this.mContext).startLeScan();
        this.mHandler.removeCallbacksAndMessages(null);
        try {
            this.mActivity.unregisterReceiver(this.mReceiver);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void onDisconnected(int i) {
        LogUtil.e("多连接 onDisconnected " + i);
        BluetoothMultitudeAdapter bluetoothMultitudeAdapter = this.mAdapter;
        if (bluetoothMultitudeAdapter != null) {
            bluetoothMultitudeAdapter.setClicked(false);
            this.mAdapter.notifyDataSetChanged();
        }
    }

    @Override // com.hismith.hismithapp.bluetooth.BleManager.OnScanDeviceListener
    public void onScanDevice(BluetoothDevice bluetoothDevice, String str, String str2) {
        String str3;
        String str4;
        String address = bluetoothDevice.getAddress();
        if (this.mBluetoothAddressList.containsKey(address) || TextUtils.isEmpty(address)) {
            str3 = str;
            str4 = str2;
            if (!TextUtils.isEmpty(str3) && (TextUtils.isEmpty(this.mBluetoothAddressList.get(address)) || !str3.equals(this.mBluetoothAddressList.get(address)))) {
                this.mBluetoothAddressList.put(address, str3);
                int i = 0;
                while (true) {
                    if (i >= this.mBluetoothList.size()) {
                        break;
                    }
                    if (this.mBluetoothList.get(i).getDeviceAddress().equals(address)) {
                        this.mBluetoothList.get(i).setDeviceName(str3);
                        this.mAdapter.notifyItemChanged(i);
                        break;
                    }
                    i++;
                }
            }
        } else {
            this.mBluetoothAddressList.put(address, str);
            str3 = str;
            str4 = str2;
            this.mBluetoothList.add(new BleDeviceModel(str3, bluetoothDevice.getAddress(), str4, bluetoothDevice, ProductManager.getProduct_v1(this.mContext, str2)));
            this.mAdapter.notifyDataSetChanged();
        }
        LogUtil.e("onScanDevice name:" + str3 + ", code:" + str4 + ", size:" + this.mBluetoothList.size());
    }

    public void sortBluetoothList() {
        Collections.sort(this.mBluetoothList, new Comparator<BleDeviceModel>() { // from class: com.hismith.hismithapp.fragment.BluetoothMultitudeFragment.5
            @Override // java.util.Comparator
            public int compare(BleDeviceModel bleDeviceModel, BleDeviceModel bleDeviceModel2) {
                BleManager bleManager = BleManager.getInstance(BluetoothMultitudeFragment.this.mContext);
                boolean z = bleManager.isConnectedBleProduct(bleDeviceModel.getDeviceAddress()) == 2;
                boolean z2 = bleManager.isConnectedBleProduct(bleDeviceModel2.getDeviceAddress()) == 2;
                if (!z || z2) {
                    return (z || !z2) ? 0 : 1;
                }
                return -1;
            }
        });
    }
}
