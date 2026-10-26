package com.hismith.hismithapp.fragment;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.graphics.drawable.Drawable;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.text.TextUtils;
import android.util.DisplayMetrics;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.view.ViewTreeObserver;
import android.view.animation.Animation;
import android.view.animation.ScaleAnimation;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.activity.result.ActivityResult;
import androidx.activity.result.ActivityResultCallback;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.FragmentActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.bumptech.glide.Glide;
import com.bumptech.glide.RequestBuilder;
import com.bumptech.glide.request.target.SimpleTarget;
import com.bumptech.glide.request.transition.Transition;
import com.facebook.appevents.internal.ViewHierarchyConstants;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.hismith.hismithapp.BluetootListActivity;
import com.hismith.hismithapp.ControlAcceptActivity2;
import com.hismith.hismithapp.ControlSponsorActivity2;
import com.hismith.hismithapp.MainActivity;
import com.hismith.hismithapp.PlayLineActivity;
import com.hismith.hismithapp.PlayModeActivity;
import com.hismith.hismithapp.PlaySoloActivity;
import com.hismith.hismithapp.PlayVideoActivity;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.RemotePlayActivity;
import com.hismith.hismithapp.adapter.PlayModeAdapter;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.bluetooth.BleManager;
import com.hismith.hismithapp.dialog.PlaySelecterDialog;
import com.hismith.hismithapp.dialog.ShareDialog;
import com.hismith.hismithapp.http.RetrofitRequest;
import com.hismith.hismithapp.http.RxSubscribe;
import com.hismith.hismithapp.http.SocketManager;
import com.hismith.hismithapp.model.PlayModeListModel;
import com.hismith.hismithapp.model.PlayModeModel;
import com.hismith.hismithapp.popup.MoreAttachPopup2;
import com.hismith.hismithapp.service.HuDongPlayService;
import com.hismith.hismithapp.utils.AppUtil;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.MeasureUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import com.hismith.hismithapp.utils.ToastUtil;
import com.lxj.xpopup.core.BasePopupView;
import com.scwang.smart.refresh.layout.SmartRefreshLayout;
import com.scwang.smart.refresh.layout.api.RefreshLayout;
import com.scwang.smart.refresh.layout.listener.OnRefreshLoadMoreListener;
import com.yhao.floatwindow.FloatWindow;
import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.schedulers.Schedulers;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import okhttp3.OkHttpClient;
import okhttp3.Response;
import okhttp3.WebSocket;

/* loaded from: classes3.dex */
public class PlayFragment extends LazyLoadFragment implements View.OnClickListener, SocketManager.OnSocketNotifyListener {
    public static final int WHAT_HIDE_TIP = 3;
    private RetrofitRequest baseRequest;
    private OkHttpClient httpClient;
    private PlayModeAdapter mAdapter;
    private ImageView mBgBattery;
    private ActivityResultLauncher mBluetoothLauncher;
    private ImageView mImgBattery;
    private ImageView mImgBluetooth;
    private ImageView mImgDevice;
    private ImageView mImgPlay;
    private ConstraintLayout mLayoutBattery;
    private LinearLayoutManager mLayoutManager;
    private List<PlayModeModel> mList;
    private PlaySelecterDialog mPlaySelecterDialog;
    private BasePopupView mPlaySelecterPopupView;
    private RecyclerView mRecyclerView;
    private SmartRefreshLayout mRefreshLayout;
    private String mRequestId;
    private TextView mTxtBluetooth;
    private View mViewBattery;
    private WebSocket mWebSocket;
    private MainActivity mainActivity;
    private MoreAttachPopup2 moreAttachPopup2;
    private String mRequestType = ViewHierarchyConstants.DIMENSION_TOP_KEY;
    private boolean isBleConnected = false;
    private Handler mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.fragment.PlayFragment.1
        @Override // android.os.Handler.Callback
        public boolean handleMessage(Message message) {
            if (message.what != 3 || PlayFragment.this.moreAttachPopup2 == null) {
                return false;
            }
            PlayFragment.this.moreAttachPopup2.dismiss();
            return false;
        }
    });
    private int animationDuration = 300;
    private BroadcastReceiver mReceiver = new BroadcastReceiver() { // from class: com.hismith.hismithapp.fragment.PlayFragment.11
        @Override // android.content.BroadcastReceiver
        public void onReceive(Context context, Intent intent) {
            if (Constans.RECEIVER_BLE_RECE_DATA.equals(intent.getAction())) {
                String stringExtra = intent.getStringExtra(Constans.BleType);
                String stringExtra2 = intent.getStringExtra(Constans.BleData0);
                intent.getStringExtra(Constans.BleData1);
                if (("CA".equals(stringExtra) || "CC".equals(stringExtra)) && "00".equals(stringExtra2)) {
                    PlayFragment.this.changeBatteryView();
                }
            }
        }
    };

    private boolean bluetoothIsEnabled() {
        if (Build.VERSION.SDK_INT >= 31) {
            return true;
        }
        if (BleManager.getInstance(this.mainActivity).getBluetoothAdapter() == null) {
            return false;
        }
        if (BleManager.getInstance(this.mainActivity).getBluetoothAdapter().isEnabled()) {
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
    public void changeBatteryView() {
        if (BleManager.getInstance(this.mainActivity).getmConnBleDevice() == null || BleManager.getInstance(this.mainActivity).getmConnBleDevice().getBattery() <= -1) {
            return;
        }
        if (this.mLayoutBattery.getVisibility() != 0) {
            LogUtil.e("电量：显示");
            this.mLayoutBattery.setVisibility(0);
        }
        this.mViewBattery.getViewTreeObserver().addOnGlobalLayoutListener(new ViewTreeObserver.OnGlobalLayoutListener() { // from class: com.hismith.hismithapp.fragment.PlayFragment.12
            @Override // android.view.ViewTreeObserver.OnGlobalLayoutListener
            public void onGlobalLayout() {
                PlayFragment.this.mViewBattery.getViewTreeObserver().removeOnGlobalLayoutListener(this);
                ViewGroup.LayoutParams layoutParams = PlayFragment.this.mImgBattery.getLayoutParams();
                LogUtil.e("电量view高度：" + PlayFragment.this.mViewBattery.getHeight());
                int height = (int) (PlayFragment.this.mViewBattery.getHeight() * (BleManager.getInstance(PlayFragment.this.mainActivity).getmConnBleDevice().getBattery() / 10.0d));
                PlayFragment playFragment = PlayFragment.this;
                if (height >= 0) {
                    playFragment.mImgBattery.setVisibility(0);
                } else {
                    playFragment.mImgBattery.setVisibility(8);
                }
                layoutParams.height = height;
                LogUtil.e("电量高度：" + layoutParams.height);
                PlayFragment.this.mImgBattery.setLayoutParams(layoutParams);
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void dissPlaySelecterDialog() {
        PlaySelecterDialog playSelecterDialog = this.mPlaySelecterDialog;
        if (playSelecterDialog != null) {
            playSelecterDialog.dissDialog();
        }
    }

    private void init() {
        String str = (String) SharedPreferencesUtil.get(this.mainActivity, Constans.mainBackgroundUrl);
        if (!TextUtils.isEmpty(str)) {
            Glide.with((FragmentActivity) this.mainActivity).load(str).into((RequestBuilder<Drawable>) new SimpleTarget<Drawable>() { // from class: com.hismith.hismithapp.fragment.PlayFragment.2
                public void onResourceReady(Drawable drawable, Transition<? super Drawable> transition) {
                    PlayFragment.this.mImgDevice.setImageDrawable(drawable);
                }

                @Override // com.bumptech.glide.request.target.Target
                public /* bridge */ /* synthetic */ void onResourceReady(Object obj, Transition transition) {
                    onResourceReady((Drawable) obj, (Transition<? super Drawable>) transition);
                }
            });
        }
        DisplayMetrics displayMetrics = new DisplayMetrics();
        this.mainActivity.getWindowManager().getDefaultDisplay().getMetrics(displayMetrics);
        SocketManager.getInstance().setScreenSize(displayMetrics.widthPixels, displayMetrics.heightPixels);
        if (BleManager.getInstance(this.mainActivity).getStatus() == 2) {
            changeBluetooth(1);
        }
        showBluetoothGuide();
        makeFilter();
    }

    private void initActivityResult() {
        this.mBluetoothLauncher = registerForActivityResult(new ActivityResultContracts.StartActivityForResult(), new ActivityResultCallback<ActivityResult>() { // from class: com.hismith.hismithapp.fragment.PlayFragment.5
            @Override // androidx.activity.result.ActivityResultCallback
            public void onActivityResult(ActivityResult activityResult) {
                if (activityResult.getResultCode() == -1) {
                    LogUtil.e("蓝牙打开");
                    PlayFragment.this.startActivity(new Intent(PlayFragment.this.mainActivity, (Class<?>) BluetootListActivity.class));
                } else if (activityResult.getResultCode() == 0) {
                    LogUtil.e("蓝牙打不开");
                }
            }
        });
    }

    private void initPlaySelecterDialog() {
        PlaySelecterDialog playSelecterDialog = new PlaySelecterDialog(this.mainActivity);
        this.mPlaySelecterDialog = playSelecterDialog;
        playSelecterDialog.setOnItemClickListener(new PlaySelecterDialog.OnItemClickListener() { // from class: com.hismith.hismithapp.fragment.PlayFragment.8
            @Override // com.hismith.hismithapp.dialog.PlaySelecterDialog.OnItemClickListener
            public void onItemClick(int i) {
                if (i == 0) {
                    if (BleManager.getInstance(PlayFragment.this.mainActivity).getStatus() != 2 || TextUtils.isEmpty(MyApplication.getApplication().getDeviceCode())) {
                        ToastUtil.showShortToastCenter(PlayFragment.this.mainActivity, LanguageUtil.getLanguageByKey(PlayFragment.this.mainActivity, "noConnectedDevice"));
                        return;
                    } else {
                        PlayFragment.this.startActivity(new Intent(PlayFragment.this.mainActivity, (Class<?>) PlaySoloActivity.class));
                        return;
                    }
                }
                if (i == 1) {
                    if (BleManager.getInstance(PlayFragment.this.mainActivity).getStatus() != 2 || TextUtils.isEmpty(MyApplication.getApplication().getDeviceCode())) {
                        ToastUtil.showShortToastCenter(PlayFragment.this.mainActivity, LanguageUtil.getLanguageByKey(PlayFragment.this.mainActivity, "noConnectedDevice"));
                        return;
                    }
                    SocketManager.getInstance().connSocket(1);
                    SocketManager.getInstance().setOnSocketNotifyListener(PlayFragment.this);
                    if (PlayFragment.this.mainActivity.progressDialog != null && !PlayFragment.this.mainActivity.progressDialog.isShowing()) {
                        PlayFragment.this.mainActivity.progressDialog.show();
                    }
                    PlayFragment.this.dissPlaySelecterDialog();
                    return;
                }
                if (i == 2) {
                    if (BleManager.getInstance(PlayFragment.this.mainActivity).getStatus() != 2 || TextUtils.isEmpty(MyApplication.getApplication().getDeviceCode())) {
                        ToastUtil.showShortToastCenter(PlayFragment.this.mainActivity, LanguageUtil.getLanguageByKey(PlayFragment.this.mainActivity, "noConnectedDevice"));
                        return;
                    } else {
                        PlayFragment.this.startActivity(new Intent(PlayFragment.this.mainActivity, (Class<?>) PlayModeActivity.class));
                        PlayFragment.this.dissPlaySelecterDialog();
                        return;
                    }
                }
                if (i == 3) {
                    if (BleManager.getInstance(PlayFragment.this.mainActivity).getStatus() != 2 || TextUtils.isEmpty(MyApplication.getApplication().getDeviceCode())) {
                        ToastUtil.showShortToastCenter(PlayFragment.this.mainActivity, LanguageUtil.getLanguageByKey(PlayFragment.this.mainActivity, "noConnectedDevice"));
                        return;
                    } else {
                        PlayFragment.this.mainActivity.startActivity(new Intent(PlayFragment.this.mainActivity, (Class<?>) PlayLineActivity.class));
                        PlayFragment.this.dissPlaySelecterDialog();
                        return;
                    }
                }
                if (i != 4) {
                    return;
                }
                if (BleManager.getInstance(PlayFragment.this.mainActivity).getStatus() != 2 || TextUtils.isEmpty(MyApplication.getApplication().getDeviceCode())) {
                    ToastUtil.showShortToastCenter(PlayFragment.this.mainActivity, LanguageUtil.getLanguageByKey(PlayFragment.this.mainActivity, "noConnectedDevice"));
                } else {
                    PlayFragment.this.mainActivity.startActivity(new Intent(PlayFragment.this.mainActivity, (Class<?>) PlayVideoActivity.class));
                    PlayFragment.this.dissPlaySelecterDialog();
                }
            }
        });
        this.mPlaySelecterDialog.setOnCloseListener(new PlaySelecterDialog.OnCloseListener() { // from class: com.hismith.hismithapp.fragment.PlayFragment.9
            @Override // com.hismith.hismithapp.dialog.PlaySelecterDialog.OnCloseListener
            public void onClose() {
                PlayFragment.this.dissPlaySelecterDialog();
            }
        });
        this.mPlaySelecterDialog.showDialog();
    }

    private void makeFilter() {
        IntentFilter intentFilter = new IntentFilter();
        intentFilter.addAction("android.bluetooth.adapter.action.STATE_CHANGED");
        intentFilter.addAction(Constans.RECEIVER_BLE_RECE_DATA);
        AppUtil.registerReceiver(this.mainActivity, this.mReceiver, intentFilter);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void playModeList(String str, int i, final String str2) {
        this.baseRequest.getApi().playModeList((String) SharedPreferencesUtil.get(this.mainActivity, Constans.Token), str, MyApplication.getApplication().getDeviceCode(), i, str2).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<PlayModeListModel>(this.mainActivity, null, false, false) { // from class: com.hismith.hismithapp.fragment.PlayFragment.10
            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onComplete() {
                super.onComplete();
                PlayFragment.this.mRefreshLayout.finishRefresh();
                PlayFragment.this.mRefreshLayout.finishLoadMore();
                if (PlayFragment.this.mAdapter != null) {
                    PlayFragment.this.mAdapter.dissMorePopup();
                }
            }

            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onError(Throwable th) {
                super.onError(th);
                PlayFragment.this.mRefreshLayout.finishRefresh();
                PlayFragment.this.mRefreshLayout.finishLoadMore();
                if (PlayFragment.this.mAdapter != null) {
                    PlayFragment.this.mAdapter.dissMorePopup();
                }
            }

            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(PlayModeListModel playModeListModel) {
                if (playModeListModel.getSuccess() == 1) {
                    if (TextUtils.isEmpty(str2)) {
                        PlayFragment.this.mList.clear();
                        PlayFragment.this.mAdapter.notifyDataSetChanged();
                    }
                    if (!TextUtils.isEmpty(playModeListModel.getRequestId())) {
                        PlayFragment.this.mRequestId = playModeListModel.getRequestId();
                    }
                    if (playModeListModel.getPlayModes() == null || playModeListModel.getPlayModes().length <= 0) {
                        return;
                    }
                    PlayFragment.this.mList.addAll(Arrays.asList(playModeListModel.getPlayModes()));
                    PlayFragment.this.mAdapter.notifyDataSetChanged();
                }
            }
        });
    }

    private void setListener() {
        this.mImgBluetooth.setOnClickListener(this);
        this.mImgPlay.setOnClickListener(this);
        this.mRefreshLayout.setOnRefreshLoadMoreListener(new OnRefreshLoadMoreListener() { // from class: com.hismith.hismithapp.fragment.PlayFragment.4
            @Override // com.scwang.smart.refresh.layout.listener.OnLoadMoreListener
            public void onLoadMore(RefreshLayout refreshLayout) {
                PlayFragment playFragment = PlayFragment.this;
                playFragment.playModeList(playFragment.mRequestType, 0, PlayFragment.this.mRequestId);
            }

            @Override // com.scwang.smart.refresh.layout.listener.OnRefreshListener
            public void onRefresh(RefreshLayout refreshLayout) {
                PlayFragment.this.mRequestId = "";
                PlayFragment playFragment = PlayFragment.this;
                playFragment.playModeList(playFragment.mRequestType, 1, PlayFragment.this.mRequestId);
            }
        });
    }

    private void showBluetoothGuide() {
        this.mImgBluetooth.getViewTreeObserver().addOnGlobalLayoutListener(new ViewTreeObserver.OnGlobalLayoutListener() { // from class: com.hismith.hismithapp.fragment.PlayFragment.3
            @Override // android.view.ViewTreeObserver.OnGlobalLayoutListener
            public void onGlobalLayout() {
                if (MyApplication.getApplication().isBluetoothShowGuide()) {
                    String languageByKey = LanguageUtil.getLanguageByKey(PlayFragment.this.mainActivity, "bluetoothNotConnected");
                    PlayFragment.this.moreAttachPopup2 = new MoreAttachPopup2(PlayFragment.this.mainActivity, languageByKey);
                    PlayFragment.this.moreAttachPopup2.showDownPop(PlayFragment.this.mImgBluetooth);
                    MyApplication.getApplication().setBluetoothShowGuide(false);
                    PlayFragment.this.mHandler.sendEmptyMessageDelayed(3, 8000L);
                }
            }
        });
    }

    private void startAnimation() {
        ScaleAnimation scaleAnimation = new ScaleAnimation(1.0f, 0.0f, 1.0f, 0.0f, 1, 0.5f, 1, 0.5f);
        scaleAnimation.setDuration(this.animationDuration);
        scaleAnimation.setFillAfter(true);
        scaleAnimation.setRepeatCount(0);
        scaleAnimation.setAnimationListener(new Animation.AnimationListener() { // from class: com.hismith.hismithapp.fragment.PlayFragment.6
            @Override // android.view.animation.Animation.AnimationListener
            public void onAnimationEnd(Animation animation) {
            }

            @Override // android.view.animation.Animation.AnimationListener
            public void onAnimationRepeat(Animation animation) {
            }

            @Override // android.view.animation.Animation.AnimationListener
            public void onAnimationStart(Animation animation) {
            }
        });
        this.mImgPlay.startAnimation(scaleAnimation);
    }

    private void stopAnimation() {
        ScaleAnimation scaleAnimation = new ScaleAnimation(0.0f, 1.0f, 0.0f, 1.0f, 1, 0.5f, 1, 0.5f);
        scaleAnimation.setDuration(this.animationDuration);
        scaleAnimation.setFillAfter(true);
        scaleAnimation.setRepeatCount(0);
        scaleAnimation.setAnimationListener(new Animation.AnimationListener() { // from class: com.hismith.hismithapp.fragment.PlayFragment.7
            @Override // android.view.animation.Animation.AnimationListener
            public void onAnimationEnd(Animation animation) {
            }

            @Override // android.view.animation.Animation.AnimationListener
            public void onAnimationRepeat(Animation animation) {
            }

            @Override // android.view.animation.Animation.AnimationListener
            public void onAnimationStart(Animation animation) {
            }
        });
        this.mImgPlay.startAnimation(scaleAnimation);
    }

    @Override // com.hismith.hismithapp.http.SocketManager.OnSocketNotifyListener
    public void OnNotify(String str) {
        if (this.mainActivity.progressDialog != null && this.mainActivity.progressDialog.isShowing()) {
            this.mainActivity.progressDialog.dismiss();
        }
        SocketManager.getInstance().setOnSocketNotifyListener(null);
        if (TextUtils.isEmpty(str)) {
            return;
        }
        try {
            JsonObject asJsonObject = new JsonParser().parse(str).getAsJsonObject();
            if (1 != asJsonObject.get("success").getAsInt() || asJsonObject.get("remoteUrl") == null) {
                return;
            }
            new ShareDialog(this.mainActivity, asJsonObject.get("remoteUrl").getAsString()).showDialog();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void autoRefreshTop() {
        LogUtil.e("刷新界面：" + MyApplication.getApplication().getDeviceCode());
        PlayModeAdapter playModeAdapter = this.mAdapter;
        if (playModeAdapter != null) {
            playModeAdapter.dissMorePopup();
        }
        SmartRefreshLayout smartRefreshLayout = this.mRefreshLayout;
        if (smartRefreshLayout != null) {
            smartRefreshLayout.autoRefresh();
        }
    }

    public void changeBluetooth(int i) {
        LogUtil.e("changeBluetooth state:" + i);
        if (BleManager.getInstance(this.mainActivity).getmConnBleDevice() == null || BleManager.getInstance(this.mainActivity).getmConnBleDevice().getBattery() != -1) {
            changeBatteryView();
        } else {
            this.mLayoutBattery.setVisibility(8);
        }
        if (i != 1) {
            this.isBleConnected = false;
            if (this.mRefreshLayout.getVisibility() != 8) {
                this.mRefreshLayout.setVisibility(8);
            }
            this.mImgBluetooth.setImageDrawable(ContextCompat.getDrawable(this.mainActivity, R.mipmap.ic_play_bluetooth_disconnect));
            this.mTxtBluetooth.setTextColor(ContextCompat.getColor(this.mainActivity, R.color.gray9));
            LanguageUtil.setText(this.mainActivity, this.mTxtBluetooth, "deviceNotConnect");
            return;
        }
        this.isBleConnected = true;
        if (FloatWindow.get(Constans.FLOAT_REMOTE_WINDOW) == null && this.mRefreshLayout.getVisibility() == 8) {
            this.mRefreshLayout.setVisibility(0);
        }
        this.mImgBluetooth.setImageDrawable(ContextCompat.getDrawable(this.mainActivity, R.mipmap.ic_play_bluetooth_connect));
        String str = (String) SharedPreferencesUtil.get(this.mainActivity, Constans.bluetoothConnName);
        this.mTxtBluetooth.setTextColor(ContextCompat.getColor(this.mainActivity, R.color.white));
        this.mTxtBluetooth.setText(str);
        MoreAttachPopup2 moreAttachPopup2 = this.moreAttachPopup2;
        if (moreAttachPopup2 != null) {
            moreAttachPopup2.dismiss();
        }
        autoRefreshTop();
    }

    @Override // com.hismith.hismithapp.fragment.LazyLoadFragment
    protected void lazyLoad() {
        init();
        setListener();
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        int id = view.getId();
        if (id == R.id.play_img_bluetooth) {
            if (bluetoothIsEnabled()) {
                startActivity(new Intent(this.mainActivity, (Class<?>) BluetootListActivity.class));
                return;
            }
            return;
        }
        if (id == R.id.play_img_play) {
            if (SocketManager.getInstance().isRemoteConnect()) {
                this.mainActivity.startActivity(new Intent(this.mainActivity, (Class<?>) RemotePlayActivity.class));
            } else if (!SocketManager.getInstance().isHuDongConnect()) {
                initPlaySelecterDialog();
            } else if (HuDongPlayService.getClientConnection() == 1) {
                Intent intent = new Intent(this.mainActivity, (Class<?>) ControlSponsorActivity2.class);
                intent.addFlags(268435456);
                startActivity(intent);
            } else if (HuDongPlayService.getServerConnection() == 1) {
                Intent intent2 = new Intent(this.mainActivity, (Class<?>) ControlAcceptActivity2.class);
                intent2.addFlags(268435456);
                startActivity(intent2);
            }
            FloatWindow.destroy(Constans.FLOAT_REMOTE_WINDOW);
        }
    }

    @Override // com.hismith.hismithapp.http.SocketManager.OnSocketNotifyListener
    public void onClosed(int i, String str) {
        if (this.mainActivity.progressDialog != null) {
            this.mainActivity.progressDialog.dismiss();
        }
    }

    @Override // com.hismith.hismithapp.http.SocketManager.OnSocketNotifyListener
    public void onCodeNull() {
        LogUtil.e("SocketManager onCodeNull()");
    }

    @Override // com.hismith.hismithapp.fragment.LazyLoadFragment, androidx.fragment.app.Fragment
    public void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        this.mainActivity = (MainActivity) getActivity();
        initActivityResult();
    }

    @Override // androidx.fragment.app.Fragment
    public View onCreateView(LayoutInflater layoutInflater, ViewGroup viewGroup, Bundle bundle) {
        View inflate = layoutInflater.inflate(R.layout.fragment_play, (ViewGroup) null);
        this.mImgBluetooth = (ImageView) inflate.findViewById(R.id.play_img_bluetooth);
        this.mTxtBluetooth = (TextView) inflate.findViewById(R.id.play_txt_bluetooth);
        this.mImgDevice = (ImageView) inflate.findViewById(R.id.play_img_device);
        this.mImgPlay = (ImageView) inflate.findViewById(R.id.play_img_play);
        this.mRefreshLayout = (SmartRefreshLayout) inflate.findViewById(R.id.play_refreshLayout);
        this.mRecyclerView = (RecyclerView) inflate.findViewById(R.id.play_recycler_view);
        this.mLayoutBattery = (ConstraintLayout) inflate.findViewById(R.id.layout_battery);
        this.mViewBattery = inflate.findViewById(R.id.view_battery);
        this.mImgBattery = (ImageView) inflate.findViewById(R.id.img_battery);
        this.mBgBattery = (ImageView) inflate.findViewById(R.id.bg_battery);
        LanguageUtil.setText(this.mainActivity, this.mTxtBluetooth, "deviceNotConnect");
        inflate.setPadding(0, MeasureUtil.getStatusBarHeight(this.mainActivity), 0, 0);
        this.baseRequest = new RetrofitRequest();
        LinearLayoutManager linearLayoutManager = new LinearLayoutManager(this.mainActivity);
        this.mLayoutManager = linearLayoutManager;
        this.mRecyclerView.setLayoutManager(linearLayoutManager);
        this.mList = new ArrayList();
        PlayModeAdapter playModeAdapter = new PlayModeAdapter(this.mainActivity, this.mList, this.mRequestType);
        this.mAdapter = playModeAdapter;
        this.mRecyclerView.setAdapter(playModeAdapter);
        return inflate;
    }

    @Override // androidx.fragment.app.Fragment
    public void onDestroy() {
        super.onDestroy();
        MoreAttachPopup2 moreAttachPopup2 = this.moreAttachPopup2;
        if (moreAttachPopup2 != null) {
            moreAttachPopup2.dismiss();
        }
        try {
            this.mainActivity.unregisterReceiver(this.mReceiver);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override // com.hismith.hismithapp.http.SocketManager.OnSocketNotifyListener
    public void onFailure(Throwable th, Response response) {
        if (this.mainActivity.progressDialog != null) {
            this.mainActivity.progressDialog.dismiss();
        }
    }

    @Override // com.hismith.hismithapp.http.SocketManager.OnSocketNotifyListener
    public void onOpen() {
    }

    @Override // androidx.fragment.app.Fragment
    public void onPause() {
        super.onPause();
        dissPlaySelecterDialog();
    }

    @Override // androidx.fragment.app.Fragment
    public void onResume() {
        super.onResume();
        if (FloatWindow.get(Constans.FLOAT_REMOTE_WINDOW) != null && FloatWindow.get(Constans.FLOAT_REMOTE_WINDOW).isShowing()) {
            this.mRefreshLayout.setVisibility(8);
        } else if (this.mRefreshLayout.getVisibility() == 8 && this.isBleConnected) {
            this.mRefreshLayout.setVisibility(0);
        }
    }
}
