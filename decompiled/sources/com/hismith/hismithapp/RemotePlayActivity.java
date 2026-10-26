package com.hismith.hismithapp;

import android.content.BroadcastReceiver;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.graphics.Point;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.text.TextUtils;
import android.view.KeyEvent;
import android.view.MotionEvent;
import android.view.View;
import android.view.animation.Animation;
import android.view.animation.AnimationUtils;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.activity.OnBackPressedCallback;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.FragmentActivity;
import androidx.fragment.app.FragmentTransaction;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.bumptech.glide.Glide;
import com.facebook.CallbackManager;
import com.facebook.FacebookCallback;
import com.facebook.FacebookException;
import com.facebook.share.Sharer;
import com.facebook.share.model.ShareLinkContent;
import com.facebook.share.widget.MessageDialog;
import com.facebook.share.widget.ShareDialog;
import com.hismith.hismithapp.adapter.RemoteChatAdapter;
import com.hismith.hismithapp.bluetooth.BleManager;
import com.hismith.hismithapp.dialog.HudongSettingDialog;
import com.hismith.hismithapp.fragment.remote.RemoteCC_SMSPlayFragment;
import com.hismith.hismithapp.fragment.remote.RemoteCC_SMVPlayFragment;
import com.hismith.hismithapp.fragment.remote.RemoteControlFragment;
import com.hismith.hismithapp.fragment.remote.RemoteSMPlayFragment;
import com.hismith.hismithapp.http.SocketManager;
import com.hismith.hismithapp.manager.RemoteDataManager;
import com.hismith.hismithapp.manager.WxShareUtils;
import com.hismith.hismithapp.model.RemoteChatModel;
import com.hismith.hismithapp.product.ProductUtil;
import com.hismith.hismithapp.receiver.NetBroadcastReceiver;
import com.hismith.hismithapp.utils.AppUtil;
import com.hismith.hismithapp.utils.CheckUtil;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.SoftKeyboardUtil;
import com.hismith.hismithapp.utils.ToastUtil;
import com.hismith.hismithapp.view.ClipLayout;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/* loaded from: classes3.dex */
public class RemotePlayActivity extends BaseActivity implements RemoteDataManager.Callback, NetBroadcastReceiver.NetChangeListener, BleManager.OnConnectStateListener {
    private static final long DOUBLE_CLICK_DELAY = 500;
    private ClipboardManager clipboard;
    private HudongSettingDialog hudongSettingDialog;
    private RemoteChatAdapter mChatAdapter;
    private List<RemoteChatModel> mChatList;
    private CallbackManager mFaceCallbackManager;
    private MessageDialog mFaceMessageDialog;
    private ShareDialog mFaceShareDialog;
    private RemoteControlFragment mFragment;
    private View mFragmentShare;
    private FrameLayout mFrameLayout;
    private ImageView mImgBg;
    private ImageView mImgCloseLeft;
    private ImageView mImgCloseRight;
    private ImageView mImgFloat;
    private ImageView mImgNetworkError;
    private ImageView mImgPiu;
    private ImageView mImgSetting;
    private LinearLayoutManager mLayoutManager;
    private NetBroadcastReceiver mNetBroadcastReceiver;
    private TextView mTitleTxtName;
    private TextView mTxtNetworkError;
    private View mViewNetworkError;
    private ClipLayout remote_chat_clip_layout;
    private EditText remote_edit_message;
    private ImageView remote_img_chat;
    private ConstraintLayout remote_layout_chat;
    private RecyclerView remote_recycler_chat;
    private TextView title_txt_status;
    private int mLastNetStatus = -1;
    private final int WHAT_SOCKET_OPEN = 1;
    private final int WHAT_SOCKET_FAILURE = 2;
    private final int WHAT_TOAST = 3;
    private final int REQUEST_SMALL = 101;
    private int leftAction = 1;
    private int rightAction = 1;
    private String remoteUrl = "";
    private boolean isShowShare = false;
    private int shareId = 0;
    private boolean isShowMessage = false;
    boolean isMessageAnimating = false;
    private Handler mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.RemotePlayActivity.1
        @Override // android.os.Handler.Callback
        public boolean handleMessage(Message message) {
            int i = message.what;
            if (i == 1) {
                RemotePlayActivity.this.mViewNetworkError.setVisibility(8);
            } else if (i == 2) {
                RemotePlayActivity remotePlayActivity = RemotePlayActivity.this;
                LanguageUtil.setText(remotePlayActivity, remotePlayActivity.mTxtNetworkError, "notNetwork");
                if (RemotePlayActivity.this.mViewNetworkError.getVisibility() == 8) {
                    RemotePlayActivity.this.mViewNetworkError.setVisibility(0);
                }
            } else if (i == 3) {
                ToastUtil.showShortToastCenter(RemotePlayActivity.this, (String) message.obj);
            }
            return false;
        }
    });
    private long lastLeftClickTime = 0;
    private long lastRightClickTime = 0;
    private View.OnTouchListener leftOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.RemotePlayActivity.10
        @Override // android.view.View.OnTouchListener
        public boolean onTouch(View view, MotionEvent motionEvent) {
            int action = motionEvent.getAction();
            if (action == 0) {
                RemotePlayActivity.this.leftAction = 0;
                RemotePlayActivity.this.exitIPlay();
            } else if (action == 1) {
                long currentTimeMillis = System.currentTimeMillis();
                long j = currentTimeMillis - RemotePlayActivity.this.lastLeftClickTime;
                RemotePlayActivity remotePlayActivity = RemotePlayActivity.this;
                if (j < 500) {
                    remotePlayActivity.exit();
                    return true;
                }
                remotePlayActivity.lastLeftClickTime = currentTimeMillis;
                RemotePlayActivity.this.leftAction = 1;
            }
            return true;
        }
    };
    private View.OnTouchListener rightOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.RemotePlayActivity.11
        @Override // android.view.View.OnTouchListener
        public boolean onTouch(View view, MotionEvent motionEvent) {
            int action = motionEvent.getAction();
            if (action == 0) {
                RemotePlayActivity.this.rightAction = 0;
                RemotePlayActivity.this.exitIPlay();
            } else if (action == 1) {
                long currentTimeMillis = System.currentTimeMillis();
                long j = currentTimeMillis - RemotePlayActivity.this.lastRightClickTime;
                RemotePlayActivity remotePlayActivity = RemotePlayActivity.this;
                if (j < 500) {
                    remotePlayActivity.exit();
                    return true;
                }
                remotePlayActivity.lastRightClickTime = currentTimeMillis;
                RemotePlayActivity.this.rightAction = 1;
            }
            return true;
        }
    };
    BroadcastReceiver mReceiver = new BroadcastReceiver() { // from class: com.hismith.hismithapp.RemotePlayActivity.12
        @Override // android.content.BroadcastReceiver
        public void onReceive(Context context, Intent intent) {
            String action = intent.getAction();
            if (Constans.RECEIVER_PRODUCT_INFO.equals(action)) {
                if ("1".equals(SocketManager.getInstance().getClientConnection())) {
                    RemotePlayActivity.this.initFragment();
                }
            } else {
                if (Constans.RECEIVER_BLE_RECE_DATA.equals(action)) {
                    LogUtil.e("蓝牙广播数据接收：" + intent.getStringExtra(Constans.BleType) + ", data0:" + intent.getStringExtra(Constans.BleData0) + ", data1:" + intent.getStringExtra(Constans.BleData1));
                    return;
                }
                if (SocketManager.RECEIVER_SOCKET_NOTIFY.equals(action)) {
                    intent.getIntExtra(SocketManager.EXTRA_SOCKET_ERROR, 0);
                    intent.getStringExtra(SocketManager.EXTRA_SOCKET_DATA);
                }
            }
        }
    };

    /* JADX INFO: Access modifiers changed from: private */
    public void addRemoteChatMessage(RemoteChatModel remoteChatModel) {
        List<RemoteChatModel> list = this.mChatList;
        if (list != null) {
            list.add(remoteChatModel);
            RemoteChatAdapter remoteChatAdapter = this.mChatAdapter;
            if (remoteChatAdapter != null) {
                remoteChatAdapter.notifyItemInserted(this.mChatList.size() - 1);
                this.mChatAdapter.notifyItemRangeChanged(this.mChatList.size() - 1, this.mChatList.size());
            }
            this.mLayoutManager.scrollToPositionWithOffset(this.mChatList.size() - 1, Integer.MIN_VALUE);
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void checkFloatPermission() {
        if (CheckUtil.checkOp(this, 24) != 0) {
            startActivityForResult(new Intent("android.settings.action.MANAGE_OVERLAY_PERMISSION", Uri.parse("package:" + getPackageName())), 101);
        } else {
            RemoteDataManager.getInstance().showFloatWindow();
            finish();
        }
    }

    private void colseSocket() {
        if (SocketManager.getInstance().getWebSocket() != null) {
            LogUtil.e("WebSocket 发送：关闭");
            SocketManager.getInstance().setOnSocketNotifyListener(null);
            SocketManager.getInstance().close();
            SocketManager.getInstance().closeSocket();
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void exit() {
        colseSocket();
        BleManager.getInstance(this).writeShutdown();
        RemoteDataManager.getInstance().release();
        this.appManager.finishActivity(this);
        startActivity(new Intent(this, (Class<?>) MainActivity.class));
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void exitIPlay() {
        if (this.leftAction == 0 && this.rightAction == 0) {
            exit();
        }
        if (this.leftAction == 1) {
            shakeAnimation(this.mImgCloseLeft);
            ToastUtil.showShortToastCenter(this, LanguageUtil.getLanguageByKey(this, "exitDoubleGameNote"));
        }
        if (this.rightAction == 1) {
            shakeAnimation(this.mImgCloseRight);
            ToastUtil.showShortToastCenter(this, LanguageUtil.getLanguageByKey(this, "exitDoubleGameNote"));
        }
    }

    private void init() {
        if (getIntent() != null) {
            this.isShowShare = getIntent().getBooleanExtra("isShowShare", false);
            this.shareId = getIntent().getIntExtra("shareId", 0);
            this.remoteUrl = getIntent().getStringExtra("remoteUrl");
        }
        LinearLayoutManager linearLayoutManager = new LinearLayoutManager(this);
        this.mLayoutManager = linearLayoutManager;
        linearLayoutManager.setStackFromEnd(true);
        this.remote_recycler_chat.setLayoutManager(this.mLayoutManager);
        this.mChatList = new ArrayList();
        RemoteChatAdapter remoteChatAdapter = new RemoteChatAdapter(this, this.mChatList);
        this.mChatAdapter = remoteChatAdapter;
        this.remote_recycler_chat.setAdapter(remoteChatAdapter);
        initChatLayout();
        boolean z = this.isShowShare;
        View view = this.mFragmentShare;
        if (z) {
            view.setVisibility(0);
        } else {
            view.setVisibility(8);
        }
        this.clipboard = (ClipboardManager) getSystemService("clipboard");
        initFaceBookShare();
        initGetRemoteUrl();
        if ("1".equals(SocketManager.getInstance().getClientConnection())) {
            initFragment();
        }
        if (!SocketManager.getInstance().isRemoteConnect()) {
            socketFailure();
        }
        if (BleManager.getInstance(this).getStatus() != 2) {
            this.title_txt_status.setVisibility(0);
        }
        registerReceiver();
        registerNetReceiver();
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void initChatLayout() {
        boolean z = this.isShowMessage;
        ImageView imageView = this.remote_img_chat;
        if (z) {
            imageView.setImageDrawable(ContextCompat.getDrawable(this, R.mipmap.ic_barrage_open));
            this.remote_chat_clip_layout.restoreFullView();
        } else {
            imageView.setImageDrawable(ContextCompat.getDrawable(this, R.mipmap.ic_barrage_close));
            this.remote_chat_clip_layout.showSingleLineAndScrollToBottom();
        }
        RemoteChatAdapter remoteChatAdapter = this.mChatAdapter;
        if (remoteChatAdapter != null) {
            remoteChatAdapter.setSingleLine(!this.isShowMessage);
        }
    }

    private void initFaceBookShare() {
        this.mFaceCallbackManager = CallbackManager.Factory.create();
        FacebookCallback<Sharer.Result> facebookCallback = new FacebookCallback<Sharer.Result>() { // from class: com.hismith.hismithapp.RemotePlayActivity.8
            @Override // com.facebook.FacebookCallback
            public void onCancel() {
                LogUtil.e("facebook share Canceled");
            }

            @Override // com.facebook.FacebookCallback
            public void onError(FacebookException facebookException) {
                LogUtil.e("facebook share " + String.format("Error: %s", facebookException.toString()));
            }

            @Override // com.facebook.FacebookCallback
            public void onSuccess(Sharer.Result result) {
                LogUtil.e("facebook share Success!");
            }
        };
        ShareDialog shareDialog = new ShareDialog(this);
        this.mFaceShareDialog = shareDialog;
        shareDialog.registerCallback(this.mFaceCallbackManager, facebookCallback);
        MessageDialog messageDialog = new MessageDialog(this);
        this.mFaceMessageDialog = messageDialog;
        messageDialog.registerCallback(this.mFaceCallbackManager, facebookCallback);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void initFragment() {
        this.mTitleTxtName.setVisibility(4);
        this.mFrameLayout.setVisibility(0);
        RemoteControlFragment remoteCC_SMSPlayFragment = ProductUtil.isSMSType() ? new RemoteCC_SMSPlayFragment() : ProductUtil.isSMVType() ? new RemoteCC_SMVPlayFragment() : !ProductUtil.isFType() ? new RemoteSMPlayFragment() : null;
        RemoteControlFragment remoteControlFragment = this.mFragment;
        if (remoteCC_SMSPlayFragment != null) {
            if (remoteControlFragment == null || remoteControlFragment.getClass() != remoteCC_SMSPlayFragment.getClass()) {
                this.mFragment = remoteCC_SMSPlayFragment;
            }
            FragmentTransaction beginTransaction = getSupportFragmentManager().beginTransaction();
            beginTransaction.replace(R.id.frame_layout, this.mFragment);
            beginTransaction.commitAllowingStateLoss();
        } else if (remoteControlFragment != null) {
            FragmentTransaction beginTransaction2 = getSupportFragmentManager().beginTransaction();
            beginTransaction2.remove(this.mFragment);
            beginTransaction2.commit();
            this.mFragment = null;
        }
        boolean hasPiuFunBle = BleManager.getInstance(this).hasPiuFunBle();
        ImageView imageView = this.mImgPiu;
        if (hasPiuFunBle) {
            imageView.setVisibility(0);
        } else {
            imageView.setVisibility(8);
        }
        if (this.remote_recycler_chat.getVisibility() == 8) {
            this.remote_recycler_chat.setVisibility(0);
        }
    }

    private void initGetRemoteUrl() {
        if (TextUtils.isEmpty(this.remoteUrl)) {
            return;
        }
        if (this.isShowShare) {
            runOnUiThread(new Runnable() { // from class: com.hismith.hismithapp.RemotePlayActivity.9
                @Override // java.lang.Runnable
                public void run() {
                    ((TextView) RemotePlayActivity.this.mFragmentShare.findViewById(R.id.friend_share_txt_content)).setText(RemotePlayActivity.this.remoteUrl);
                }
            });
            return;
        }
        int i = this.shareId;
        if (i == 5) {
            this.clipboard.setPrimaryClip(ClipData.newPlainText(null, this.remoteUrl));
            String languageByKey = LanguageUtil.getLanguageByKey(this, "copied");
            Message message = new Message();
            message.what = 3;
            message.obj = languageByKey;
            this.mHandler.sendMessage(message);
            return;
        }
        if (i == 4) {
            WxShareUtils.shareText(this, this.remoteUrl);
            return;
        }
        if (i == 1) {
            ShareLinkContent build = new ShareLinkContent.Builder().setContentUrl(Uri.parse(this.remoteUrl)).build();
            ShareDialog shareDialog = this.mFaceShareDialog;
            if (shareDialog != null) {
                shareDialog.show(build);
                return;
            }
            return;
        }
        if (i == 7) {
            ShareLinkContent build2 = new ShareLinkContent.Builder().setContentUrl(Uri.parse(this.remoteUrl)).build();
            LogUtil.e("facebook canShow:" + this.mFaceMessageDialog.canShow((MessageDialog) build2));
            MessageDialog messageDialog = this.mFaceMessageDialog;
            if (messageDialog == null || !messageDialog.canShow((MessageDialog) build2)) {
                return;
            }
            this.mFaceMessageDialog.show(build2);
            return;
        }
        if (i == 2) {
            Intent intent = new Intent();
            intent.setAction("android.intent.action.VIEW");
            intent.setData(Uri.parse("line://msg/text/" + this.remoteUrl));
            startActivity(intent);
            return;
        }
        if (i == 3) {
            Intent intent2 = new Intent();
            intent2.setAction("android.intent.action.VIEW");
            intent2.setData(Uri.parse("whatsapp://send?text=" + this.remoteUrl));
            startActivity(intent2);
        }
    }

    private void initView() {
        View findViewById = findViewById(R.id.view_network_error);
        this.mViewNetworkError = findViewById;
        this.mImgNetworkError = (ImageView) findViewById.findViewById(R.id.img_network_error);
        this.mTxtNetworkError = (TextView) this.mViewNetworkError.findViewById(R.id.txt_network_error);
        this.title_txt_status = (TextView) findViewById(R.id.title_txt_status);
        this.mImgBg = (ImageView) findViewById(R.id.remote_play_img_bg);
        TextView textView = (TextView) findViewById(R.id.title_txt_name);
        this.mTitleTxtName = textView;
        textView.setSelected(true);
        this.mImgSetting = (ImageView) findViewById(R.id.img_setting);
        this.mImgFloat = (ImageView) findViewById(R.id.remote_play_img_float);
        this.mImgCloseLeft = (ImageView) findViewById(R.id.remote_play_img_close_left);
        this.mImgCloseRight = (ImageView) findViewById(R.id.remote_play_img_close_right);
        this.mFrameLayout = (FrameLayout) findViewById(R.id.frame_layout);
        this.mImgPiu = (ImageView) findViewById(R.id.remote_play_img_piu);
        this.mFragmentShare = findViewById(R.id.remote_play_fragment_share);
        this.remote_layout_chat = (ConstraintLayout) findViewById(R.id.remote_layout_chat);
        this.remote_img_chat = (ImageView) findViewById(R.id.remote_img_chat);
        this.remote_edit_message = (EditText) findViewById(R.id.remote_edit_message);
        this.remote_chat_clip_layout = (ClipLayout) findViewById(R.id.remote_chat_clip_layout);
        this.remote_recycler_chat = (RecyclerView) findViewById(R.id.remote_recycler_chat);
        LanguageUtil.setText(this, this.mTxtNetworkError, "notNetwork");
        LanguageUtil.setText(this, this.title_txt_status, "bluetoothDisconnected");
        LanguageUtil.setText(this, this.mTitleTxtName, "waitingRemoteConnection");
        Glide.with((FragmentActivity) this).load(Integer.valueOf(R.mipmap.bg_iplay)).into(this.mImgBg);
    }

    private void registerNetReceiver() {
        IntentFilter intentFilter = new IntentFilter();
        intentFilter.addAction("android.net.conn.CONNECTIVITY_CHANGE");
        NetBroadcastReceiver netBroadcastReceiver = new NetBroadcastReceiver();
        this.mNetBroadcastReceiver = netBroadcastReceiver;
        netBroadcastReceiver.setNetChangeListener(this);
        AppUtil.registerReceiver(this, this.mNetBroadcastReceiver, intentFilter);
    }

    private void registerReceiver() {
        IntentFilter intentFilter = new IntentFilter();
        intentFilter.addAction(Constans.RECEIVER_PRODUCT_INFO);
        intentFilter.addAction(Constans.RECEIVER_BLE_RECE_DATA);
        intentFilter.addAction(SocketManager.RECEIVER_SOCKET_NOTIFY);
        AppUtil.registerReceiver(this, this.mReceiver, intentFilter);
    }

    private void remoteFragment() {
        BleManager.getInstance(this).writeShutdown();
        FragmentTransaction beginTransaction = getSupportFragmentManager().beginTransaction();
        RemoteControlFragment remoteControlFragment = this.mFragment;
        if (remoteControlFragment != null) {
            beginTransaction.remove(remoteControlFragment);
            beginTransaction.commitAllowingStateLoss();
            this.mFragment = null;
        }
        this.mTitleTxtName.setVisibility(0);
        this.mFrameLayout.setVisibility(8);
        if (this.mImgPiu.getVisibility() == 0) {
            this.mImgPiu.setVisibility(8);
        }
        if (this.remote_recycler_chat.getVisibility() == 0) {
            this.remote_recycler_chat.setVisibility(8);
        }
    }

    private void setListener() {
        RemoteDataManager.getInstance().initListener();
        RemoteDataManager.getInstance().setCallback(this);
        BleManager.getInstance(this).setOnConnectStateListener(this);
        this.mImgCloseLeft.setOnTouchListener(this.leftOnTouch);
        this.mImgCloseRight.setOnTouchListener(this.rightOnTouch);
        this.mImgSetting.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.RemotePlayActivity.3
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                if (RemotePlayActivity.this.hudongSettingDialog != null) {
                    RemotePlayActivity.this.hudongSettingDialog.dismiss();
                }
                RemotePlayActivity.this.hudongSettingDialog = new HudongSettingDialog(RemotePlayActivity.this, 100, false, 100, null);
                RemotePlayActivity.this.hudongSettingDialog.show();
            }
        });
        this.mImgFloat.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.RemotePlayActivity.4
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                RemotePlayActivity.this.checkFloatPermission();
            }
        });
        this.mViewNetworkError.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.RemotePlayActivity.5
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                if (RemotePlayActivity.this.mLastNetStatus > 0) {
                    SocketManager.getInstance().connSocket(3);
                    RemotePlayActivity remotePlayActivity = RemotePlayActivity.this;
                    LanguageUtil.setText(remotePlayActivity, remotePlayActivity.mTxtNetworkError, "isReconnecting");
                }
            }
        });
        this.remote_img_chat.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.RemotePlayActivity.6
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                RemotePlayActivity.this.isShowMessage = !r2.isShowMessage;
                RemotePlayActivity.this.initChatLayout();
            }
        });
        this.remote_edit_message.setOnEditorActionListener(new TextView.OnEditorActionListener() { // from class: com.hismith.hismithapp.RemotePlayActivity.7
            @Override // android.widget.TextView.OnEditorActionListener
            public boolean onEditorAction(TextView textView, int i, KeyEvent keyEvent) {
                if (i != 4) {
                    return false;
                }
                String obj = textView.getText().toString();
                if (obj.length() > 512) {
                    return true;
                }
                textView.setText("");
                if (!TextUtils.isEmpty(obj)) {
                    SocketManager.getInstance().chatMessage(obj);
                    RemotePlayActivity.this.addRemoteChatMessage(new RemoteChatModel(obj, 1));
                }
                SoftKeyboardUtil.hideSoftKeyboard(RemotePlayActivity.this, textView);
                return true;
            }
        });
    }

    private void socketFailure() {
        LogUtil.e("显示界面：" + this.mViewNetworkError.getVisibility());
        this.mHandler.sendEmptyMessage(2);
    }

    private void socketOpen() {
        LogUtil.e("隐藏界面：" + this.mViewNetworkError.getVisibility());
        if (this.mViewNetworkError.getVisibility() == 0) {
            this.mHandler.sendEmptyMessage(1);
        }
    }

    private void unregisterReceiver() {
        try {
            unregisterReceiver(this.mReceiver);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override // com.hismith.hismithapp.manager.RemoteDataManager.Callback
    public void onAutoUI(int i) {
        RemoteControlFragment remoteControlFragment = this.mFragment;
        if (remoteControlFragment != null) {
            remoteControlFragment.refreshAutoUI(i);
        }
    }

    @Override // com.hismith.hismithapp.receiver.NetBroadcastReceiver.NetChangeListener
    public void onChangeListener(int i) {
        if (i == 2) {
            LogUtil.e("SocketManager 网络状态：WIFI");
            int i2 = this.mLastNetStatus;
            if (i2 != -1 && i2 != 2) {
                SocketManager.getInstance().connSocket(3);
                LanguageUtil.setText(this, this.mTxtNetworkError, "isReconnecting");
            }
            this.mLastNetStatus = 2;
            return;
        }
        if (i != 1) {
            if (i == 0) {
                LogUtil.e("SocketManager 网络状态：没网络");
                if (this.mLastNetStatus == -1) {
                    socketFailure();
                }
                this.mLastNetStatus = 0;
                return;
            }
            return;
        }
        LogUtil.e("SocketManager 网络状态：移动网络");
        int i3 = this.mLastNetStatus;
        if (i3 != -1 && i3 != 1) {
            SocketManager.getInstance().connSocket(3);
            LanguageUtil.setText(this, this.mTxtNetworkError, "isReconnecting");
        }
        this.mLastNetStatus = 1;
    }

    @Override // com.hismith.hismithapp.manager.RemoteDataManager.Callback
    public void onClientConnection() {
        initFragment();
    }

    @Override // com.hismith.hismithapp.manager.RemoteDataManager.Callback
    public void onClientOffline() {
        remoteFragment();
    }

    @Override // com.hismith.hismithapp.manager.RemoteDataManager.Callback
    public void onClose() {
        socketFailure();
    }

    @Override // com.hismith.hismithapp.manager.RemoteDataManager.Callback
    public void onCodeNull() {
    }

    @Override // com.hismith.hismithapp.bluetooth.BleManager.OnConnectStateListener
    public void onConnected(int i) {
        if (i > 0) {
            runOnUiThread(new Runnable() { // from class: com.hismith.hismithapp.RemotePlayActivity.13
                @Override // java.lang.Runnable
                public void run() {
                    RemotePlayActivity.this.title_txt_status.setVisibility(8);
                }
            });
        }
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, androidx.core.app.ComponentActivity, android.app.Activity
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        getOnBackPressedDispatcher().addCallback(this, new OnBackPressedCallback(true) { // from class: com.hismith.hismithapp.RemotePlayActivity.2
            @Override // androidx.activity.OnBackPressedCallback
            public void handleOnBackPressed() {
                if (RemotePlayActivity.this.isShowShare) {
                    RemotePlayActivity.this.finish();
                }
            }
        });
        getWindow().addFlags(128);
        setContentView(R.layout.activity_remote_play);
        initView();
        init();
        setListener();
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.appcompat.app.AppCompatActivity, androidx.fragment.app.FragmentActivity, android.app.Activity
    protected void onDestroy() {
        super.onDestroy();
        this.mHandler.removeCallbacksAndMessages(null);
        try {
            NetBroadcastReceiver netBroadcastReceiver = this.mNetBroadcastReceiver;
            if (netBroadcastReceiver != null) {
                unregisterReceiver(netBroadcastReceiver);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        unregisterReceiver();
        ToastUtil.cancelToast();
    }

    @Override // com.hismith.hismithapp.bluetooth.BleManager.OnConnectStateListener
    public void onDisconnected(int i) {
        if (i == 0) {
            runOnUiThread(new Runnable() { // from class: com.hismith.hismithapp.RemotePlayActivity.14
                @Override // java.lang.Runnable
                public void run() {
                    RemotePlayActivity.this.title_txt_status.setVisibility(0);
                }
            });
        }
    }

    @Override // com.hismith.hismithapp.manager.RemoteDataManager.Callback
    public void onEndPositionUI(int i) {
        RemoteControlFragment remoteControlFragment = this.mFragment;
        if (remoteControlFragment != null) {
            remoteControlFragment.refreshStopPositionUi(i);
        }
    }

    @Override // com.hismith.hismithapp.manager.RemoteDataManager.Callback
    public void onLinePoints(Map<String, Point> map) {
        RemoteControlFragment remoteControlFragment = this.mFragment;
        if (remoteControlFragment != null) {
            remoteControlFragment.refreshLinePoints(map);
        }
    }

    @Override // com.hismith.hismithapp.manager.RemoteDataManager.Callback
    public void onMessage(String str) {
        addRemoteChatMessage(new RemoteChatModel(str, 0));
    }

    @Override // com.hismith.hismithapp.manager.RemoteDataManager.Callback
    public void onNotify() {
        socketOpen();
    }

    @Override // com.hismith.hismithapp.manager.RemoteDataManager.Callback
    public void onOpen() {
        socketOpen();
    }

    @Override // com.hismith.hismithapp.manager.RemoteDataManager.Callback
    public void onOpenClimax() {
        RemoteControlFragment remoteControlFragment = this.mFragment;
        if (remoteControlFragment != null) {
            remoteControlFragment.startHeart();
        }
    }

    @Override // com.hismith.hismithapp.manager.RemoteDataManager.Callback
    public void onPiuUI(int i) {
        ImageView imageView = this.mImgPiu;
        if (i == 1) {
            imageView.setImageResource(R.mipmap.ic_iplay_piu_select);
        } else {
            imageView.setImageResource(R.mipmap.ic_iplay_piu);
        }
    }

    @Override // com.hismith.hismithapp.manager.RemoteDataManager.Callback
    public void onPositionUI(int i) {
        RemoteControlFragment remoteControlFragment = this.mFragment;
        if (remoteControlFragment != null) {
            remoteControlFragment.refreshPositionUI(i);
        }
    }

    @Override // com.hismith.hismithapp.manager.RemoteDataManager.Callback
    public void onReversalRotation(int i) {
        RemoteControlFragment remoteControlFragment = this.mFragment;
        if (remoteControlFragment != null) {
            remoteControlFragment.refreshReversalRotation(i);
        }
    }

    @Override // com.hismith.hismithapp.manager.RemoteDataManager.Callback
    public void onSmoothUI(int i) {
        RemoteControlFragment remoteControlFragment = this.mFragment;
        if (remoteControlFragment != null) {
            remoteControlFragment.refreshSmoothnessUI(i);
        }
    }

    @Override // com.hismith.hismithapp.manager.RemoteDataManager.Callback
    public void onSpeedModeUI(int i) {
        RemoteControlFragment remoteControlFragment = this.mFragment;
        if (remoteControlFragment != null) {
            remoteControlFragment.refreshSpeedModeUI(i);
        }
    }

    @Override // com.hismith.hismithapp.manager.RemoteDataManager.Callback
    public void onSpeedUI(int i) {
        RemoteControlFragment remoteControlFragment = this.mFragment;
        if (remoteControlFragment != null) {
            remoteControlFragment.refreshSpeedUI(i);
        }
    }

    @Override // com.hismith.hismithapp.manager.RemoteDataManager.Callback
    public void onStartPositionUI(int i) {
        RemoteControlFragment remoteControlFragment = this.mFragment;
        if (remoteControlFragment != null) {
            remoteControlFragment.refreshStartPositionUI(i);
        }
    }

    @Override // com.hismith.hismithapp.manager.RemoteDataManager.Callback
    public void onVibrateLinePoints(Map<String, Point> map) {
        RemoteControlFragment remoteControlFragment = this.mFragment;
        if (remoteControlFragment != null) {
            remoteControlFragment.refreshVibrateLinePoints(map);
        }
    }

    @Override // com.hismith.hismithapp.manager.RemoteDataManager.Callback
    public void onVibrateModeUI(int i) {
        RemoteControlFragment remoteControlFragment = this.mFragment;
        if (remoteControlFragment != null) {
            remoteControlFragment.refreshVibrateModeUI(i);
        }
    }

    @Override // com.hismith.hismithapp.manager.RemoteDataManager.Callback
    public void onVibrateUI(int i) {
        RemoteControlFragment remoteControlFragment = this.mFragment;
        if (remoteControlFragment != null) {
            remoteControlFragment.refreshVibrateUI(i);
        }
    }

    @Override // com.hismith.hismithapp.manager.RemoteDataManager.Callback
    public void onWebSpeedUI(int i) {
        RemoteControlFragment remoteControlFragment = this.mFragment;
        if (remoteControlFragment != null) {
            remoteControlFragment.refreshWebSpeedUI(i);
        }
    }

    @Override // com.hismith.hismithapp.manager.RemoteDataManager.Callback
    public void onWebVibrateUI(int i) {
        RemoteControlFragment remoteControlFragment = this.mFragment;
        if (remoteControlFragment != null) {
            remoteControlFragment.refreshWebVibrateUI(i);
        }
    }

    public void shakeAnimation(View view) {
        view.clearAnimation();
        Animation loadAnimation = AnimationUtils.loadAnimation(this, R.anim.iplay_exit_anim);
        loadAnimation.reset();
        loadAnimation.setFillAfter(true);
        view.startAnimation(loadAnimation);
    }
}
