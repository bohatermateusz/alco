package com.hismith.hismithapp.service;

import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.os.Binder;
import android.os.Build;
import android.os.Handler;
import android.os.IBinder;
import android.os.Message;
import android.text.TextUtils;
import androidx.core.app.NotificationCompat;
import androidx.media3.common.C;
import androidx.media3.extractor.MpegAudioUtil;
import com.facebook.appevents.internal.ViewHierarchyConstants;
import com.facebook.internal.NativeProtocol;
import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.hismith.hismithapp.BaseActivity;
import com.hismith.hismithapp.LoginActivity;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.RemotePlayActivity;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.bluetooth.BleManager;
import com.hismith.hismithapp.dialog.ShareDialog;
import com.hismith.hismithapp.http.SocketManager;
import com.hismith.hismithapp.http.SocketSendModel;
import com.hismith.hismithapp.http.XDns;
import com.hismith.hismithapp.model.RemotePlayModel;
import com.hismith.hismithapp.utils.AppManager;
import com.hismith.hismithapp.utils.AppUtil;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import com.hismith.hismithapp.utils.ToastUtil;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.HashMap;
import java.util.concurrent.TimeUnit;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;
import okhttp3.WebSocket;
import okhttp3.WebSocketListener;
import okio.ByteString;

/* loaded from: classes3.dex */
public class RemotePlayService extends Service {
    private static final String CHANNEL_ID = "RemotePlayServiceChannel";
    public static final int SOCKET_HEARTBEAT = 6;
    public static final int TYPE_CONN_REMOTE = 1;
    public static final int TYPE_CONN_REREMOTE = 3;
    private static WebSocket mWebSocket;
    private String clientConnection;
    private String gameToken;
    private OkHttpClient httpClient;
    private RemotePlayBinder mBinder;
    private Gson mGson;
    private String remoteUrl;
    private final int timeout = 90000;
    private final int mHeartBeatDelay = MpegAudioUtil.MAX_RATE_BYTES_PER_SECOND;
    private int mWidth = 0;
    private int mHeight = 0;
    private int connType = 0;
    private boolean isError = false;
    private Handler mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.service.RemotePlayService.1
        @Override // android.os.Handler.Callback
        public boolean handleMessage(Message message) {
            if (message.what != 6) {
                return false;
            }
            RemotePlayService.this.heartbeat();
            RemotePlayService.this.mHandler.sendEmptyMessageDelayed(6, 40000L);
            return false;
        }
    });
    private BroadcastReceiver mReceiver = new BroadcastReceiver() { // from class: com.hismith.hismithapp.service.RemotePlayService.3
        @Override // android.content.BroadcastReceiver
        public void onReceive(Context context, Intent intent) {
            String action = intent.getAction();
            if (Constans.RECEIVER_PRODUCT_INFO.equals(action)) {
                return;
            }
            if (Constans.RECEIVER_BLE_RECE_DATA.equals(action)) {
                LogUtil.e("蓝牙广播数据接收：" + intent.getStringExtra(Constans.BleType) + ", data0:" + intent.getStringExtra(Constans.BleData0) + ", data1:" + intent.getStringExtra(Constans.BleData1));
                return;
            }
            if (SocketManager.RECEIVER_SOCKET_NOTIFY.equals(action)) {
                int intExtra = intent.getIntExtra(SocketManager.EXTRA_SOCKET_ERROR, -1);
                String stringExtra = intent.getStringExtra(SocketManager.EXTRA_SOCKET_DATA);
                RemotePlayService remotePlayService = RemotePlayService.this;
                if (intExtra == 1) {
                    remotePlayService.isError = true;
                } else {
                    remotePlayService.isError = false;
                    RemotePlayService.this.OnNotify(stringExtra);
                }
            }
        }
    };

    public class RemotePlayBinder extends Binder {
        public RemotePlayBinder() {
        }
    }

    private void connSocket(int i) {
        this.connType = i;
        this.httpClient = new OkHttpClient.Builder().connectTimeout(0L, TimeUnit.MILLISECONDS).dns(new XDns(90000L)).build();
        LogUtil.e("connSocket:ws://www.hismiths.com:8886");
        this.httpClient.newWebSocket(new Request.Builder().url("ws://www.hismiths.com:8886").build(), createListener());
        LogUtil.e("WebSocket 连接Socket");
    }

    private WebSocketListener createListener() {
        return new WebSocketListener() { // from class: com.hismith.hismithapp.service.RemotePlayService.2
            @Override // okhttp3.WebSocketListener
            public void onClosed(WebSocket webSocket, int i, String str) {
                super.onClosed(webSocket, i, str);
                LogUtil.e("WebSocket: onClosed code:" + i + ", reason:" + str);
            }

            @Override // okhttp3.WebSocketListener
            public void onClosing(WebSocket webSocket, int i, String str) {
                super.onClosing(webSocket, i, str);
                LogUtil.e("onClosing code:" + i);
            }

            @Override // okhttp3.WebSocketListener
            public void onFailure(WebSocket webSocket, Throwable th, Response response) {
                super.onFailure(webSocket, th, response);
                LogUtil.e("WebSocket: onFailure：" + RemotePlayService.this.printThrowable(th) + ", webSocket:" + webSocket + ", mWebSocket:" + RemotePlayService.mWebSocket + ", 是否和当前webSocket相等" + (webSocket == RemotePlayService.mWebSocket));
            }

            @Override // okhttp3.WebSocketListener
            public void onMessage(WebSocket webSocket, String str) {
                super.onMessage(webSocket, str);
                LogUtil.e("onMessage text:" + str);
                RemotePlayService.this.OnNotify(str);
            }

            @Override // okhttp3.WebSocketListener
            public void onMessage(WebSocket webSocket, ByteString byteString) {
                super.onMessage(webSocket, byteString);
                LogUtil.e("onMessage ByteString:" + byteString.toString());
            }

            @Override // okhttp3.WebSocketListener
            public void onOpen(WebSocket webSocket, Response response) {
                super.onOpen(webSocket, response);
                LogUtil.e("onOpen connType:" + RemotePlayService.this.connType);
                WebSocket unused = RemotePlayService.mWebSocket = webSocket;
                int i = RemotePlayService.this.connType;
                RemotePlayService remotePlayService = RemotePlayService.this;
                if (i == 1) {
                    remotePlayService.getRemoteUrl();
                } else if (remotePlayService.connType == 3) {
                    RemotePlayService.this.getReRemote();
                }
            }
        };
    }

    private void createNotificationChannel() {
        NotificationChannel notificationChannel = new NotificationChannel(CHANNEL_ID, "My Service Channel", 3);
        NotificationManager notificationManager = (NotificationManager) getSystemService(NotificationManager.class);
        if (notificationManager != null) {
            notificationManager.createNotificationChannel(notificationChannel);
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public String printThrowable(Throwable th) {
        StringWriter stringWriter = new StringWriter();
        PrintWriter printWriter = new PrintWriter(stringWriter);
        th.printStackTrace(printWriter);
        for (Throwable cause = th.getCause(); cause != null; cause = cause.getCause()) {
            cause.printStackTrace(printWriter);
        }
        printWriter.close();
        return stringWriter.toString();
    }

    private void receResponse(String str) {
        String str2;
        if (TextUtils.isEmpty(str)) {
            return;
        }
        LogUtil.e("WebSocket " + mWebSocket + " : 接收:" + str);
        try {
            JsonObject asJsonObject = new JsonParser().parse(str).getAsJsonObject();
            int asInt = asJsonObject.get("success") == null ? 0 : asJsonObject.get("success").getAsInt();
            str2 = "";
            if (asInt != 0) {
                if (1 == asInt) {
                    str2 = asJsonObject.get(NativeProtocol.WEB_DIALOG_ACTION) != null ? asJsonObject.get(NativeProtocol.WEB_DIALOG_ACTION).getAsString() : "";
                    if (asJsonObject.get("gameToken") != null) {
                        this.gameToken = asJsonObject.get("gameToken").getAsString();
                    }
                    if (asJsonObject.get("remoteUrl") != null) {
                        this.remoteUrl = asJsonObject.get("remoteUrl").getAsString();
                        return;
                    }
                    if (asJsonObject.get("clientConnection") != null) {
                        this.clientConnection = asJsonObject.get("clientConnection").getAsString();
                        return;
                    }
                    if (asJsonObject.get("offline") != null) {
                        this.clientConnection = "0";
                        return;
                    }
                    if ("onlinePlay".equals(str2)) {
                        if (asJsonObject.get("userRoomId") != null) {
                            r13 = asJsonObject.get("userRoomId").getAsInt();
                        }
                        if (r13 > 0) {
                            SocketManager.getInstance().enterRoom(r13);
                            return;
                        }
                        return;
                    }
                    return;
                }
                return;
            }
            String asString = asJsonObject.get("errorCode") == null ? "" : asJsonObject.get("errorCode").getAsString();
            if (asJsonObject.get("errorMessage") != null) {
                str2 = asJsonObject.get("errorMessage").getAsString();
            }
            if (TextUtils.isEmpty(asString)) {
                return;
            }
            if (!TextUtils.isEmpty(str2)) {
                ToastUtil.showShortToastCenter(this, str2);
            }
            if (Constans.errorRelogin.equals(asString)) {
                String str3 = (String) SharedPreferencesUtil.get(this, Constans.Language);
                Object obj = SharedPreferencesUtil.get(this, Constans.contactUserId, 0);
                r13 = obj != null ? ((Integer) obj).intValue() : 0;
                String str4 = (String) SharedPreferencesUtil.get(this, Constans.storeUrl);
                String str5 = (String) SharedPreferencesUtil.get(this, Constans.roomVoice);
                String str6 = (String) SharedPreferencesUtil.get(this, Constans.tempToken);
                String str7 = (String) SharedPreferencesUtil.get(this, Constans.aboutApp);
                int i = r13;
                String str8 = (String) SharedPreferencesUtil.get(this, Constans.copyright);
                String str9 = (String) SharedPreferencesUtil.get(this, Constans.mainBackgroundUrl);
                String str10 = (String) SharedPreferencesUtil.get(this, Constans.bluetoothConnAddress);
                String str11 = (String) SharedPreferencesUtil.get(this, Constans.ProductAllData);
                MyApplication.getApplication().setmUserModel(null);
                SharedPreferencesUtil.clear(this);
                SharedPreferencesUtil.put(this, Constans.Language, str3);
                SharedPreferencesUtil.put(this, Constans.contactUserId, Integer.valueOf(i));
                SharedPreferencesUtil.put(this, Constans.storeUrl, str4);
                SharedPreferencesUtil.put(this, Constans.roomVoice, str5);
                SharedPreferencesUtil.put(this, Constans.tempToken, str6);
                SharedPreferencesUtil.put(this, Constans.aboutApp, str7);
                SharedPreferencesUtil.put(this, Constans.copyright, str8);
                SharedPreferencesUtil.put(this, Constans.mainBackgroundUrl, str9);
                SharedPreferencesUtil.put(this, Constans.bluetoothConnAddress, str10);
                SharedPreferencesUtil.put(this, Constans.ProductAllData, str11);
                try {
                    Intent intent = new Intent(this, (Class<?>) LoginActivity.class);
                    intent.addFlags(268435456);
                    startActivity(intent);
                    AppManager.getInstance().finishAllOtherActivity(LoginActivity.class);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        } catch (Exception e2) {
            e2.printStackTrace();
        }
    }

    private void registerReceiver() {
        IntentFilter intentFilter = new IntentFilter();
        intentFilter.addAction(Constans.RECEIVER_PRODUCT_INFO);
        intentFilter.addAction(Constans.RECEIVER_BLE_RECE_DATA);
        intentFilter.addAction(SocketManager.RECEIVER_SOCKET_NOTIFY);
        AppUtil.registerReceiver(this, this.mReceiver, intentFilter);
        LogUtil.e("registerReceiver()");
    }

    public static void startService(Context context, int i, int i2) {
        stopService(context);
        Intent intent = new Intent(context, (Class<?>) RemotePlayService.class);
        intent.putExtra(ViewHierarchyConstants.DIMENSION_WIDTH_KEY, i);
        intent.putExtra(ViewHierarchyConstants.DIMENSION_HEIGHT_KEY, i2);
        context.startForegroundService(intent);
    }

    public static void stopService(Context context) {
        context.stopService(new Intent(context, (Class<?>) RemotePlayService.class));
    }

    private void unregisterReceiver() {
        try {
            unregisterReceiver(this.mReceiver);
            LogUtil.e("unregisterReceiver()");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void OnNotify(String str) {
        if (((BaseActivity) AppManager.getInstance().currentActivity()).progressDialog != null && ((BaseActivity) AppManager.getInstance().currentActivity()).progressDialog.isShowing()) {
            ((BaseActivity) AppManager.getInstance().currentActivity()).progressDialog.dismiss();
        }
        if (TextUtils.isEmpty(str)) {
            return;
        }
        try {
            RemotePlayModel remotePlayModel = (RemotePlayModel) this.mGson.fromJson(str, RemotePlayModel.class);
            if (1 != remotePlayModel.getSuccess() || TextUtils.isEmpty(remotePlayModel.getRemoteUrl())) {
                return;
            }
            new ShareDialog((BaseActivity) AppManager.getInstance().currentActivity(), remotePlayModel.getRemoteUrl()).showDialog();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /* JADX WARN: Multi-variable type inference failed */
    public void getReRemote() {
        String str = (String) SharedPreferencesUtil.get(this, Constans.Token);
        String deviceCode = MyApplication.getApplication().getDeviceCode();
        String deviceName = MyApplication.getApplication().getDeviceName();
        HashMap hashMap = new HashMap();
        hashMap.put(Constans.Token, str);
        hashMap.put("appBrand", "");
        hashMap.put("gameToken", this.gameToken);
        hashMap.put(NativeProtocol.WEB_DIALOG_ACTION, "getRemoteUrl");
        hashMap.put("productId", deviceCode);
        hashMap.put("productName", deviceName);
        hashMap.put(ViewHierarchyConstants.DIMENSION_WIDTH_KEY, Integer.valueOf(this.mWidth));
        hashMap.put(ViewHierarchyConstants.DIMENSION_HEIGHT_KEY, Integer.valueOf(this.mHeight));
        Object obj = SharedPreferencesUtil.get(this, Constans.RemoteShareMode, true);
        hashMap.put("modeStatus", Integer.valueOf(obj != null ? ((Boolean) obj).booleanValue() : 1));
        if (BleManager.getInstance(this).getmConnBleDevice() != null) {
            int speed = BleManager.getInstance(this).getmConnBleDevice().getSpeed();
            int mode = BleManager.getInstance(this).getmConnBleDevice().getMode();
            int vibration = BleManager.getInstance(this).getmConnBleDevice().getVibration();
            int vibrationMode = BleManager.getInstance(this).getmConnBleDevice().getVibrationMode();
            int runMode = BleManager.getInstance(this).getmConnBleDevice().getRunMode();
            int startPosition = BleManager.getInstance(this).getmConnBleDevice().getStartPosition();
            int stopPosition = BleManager.getInstance(this).getmConnBleDevice().getStopPosition();
            int position = BleManager.getInstance(this).getmConnBleDevice().getPosition();
            int smoothness = BleManager.getInstance(this).getmConnBleDevice().getSmoothness();
            int reversal = BleManager.getInstance(this).getmConnBleDevice().getReversal();
            hashMap.put("speed", Integer.valueOf(speed));
            hashMap.put("mode", Integer.valueOf(mode));
            hashMap.put("vspeed", Integer.valueOf(vibration));
            hashMap.put("vibration", Integer.valueOf(vibrationMode));
            hashMap.put("strokeType", Integer.valueOf(runMode));
            hashMap.put("strokeMin", Integer.valueOf(startPosition));
            hashMap.put("strokeMax", Integer.valueOf(stopPosition));
            hashMap.put("strokeNow", Integer.valueOf(position));
            hashMap.put("smooth", Integer.valueOf(smoothness));
            hashMap.put("reversal", Integer.valueOf(reversal));
        }
        send(this.mGson.toJson(hashMap));
    }

    public void getRemoteUrl() {
        String str = (String) SharedPreferencesUtil.get(this, Constans.Token);
        String deviceCode = BleManager.getInstance(this).getmConnBleDevice().getDeviceCode();
        String deviceName = BleManager.getInstance(this).getmConnBleDevice().getDeviceName();
        int speed = BleManager.getInstance(this).getmConnBleDevice().getSpeed();
        int mode = BleManager.getInstance(this).getmConnBleDevice().getMode();
        int vibration = BleManager.getInstance(this).getmConnBleDevice().getVibration();
        int vibrationMode = BleManager.getInstance(this).getmConnBleDevice().getVibrationMode();
        int runMode = BleManager.getInstance(this).getmConnBleDevice().getRunMode();
        int startPosition = BleManager.getInstance(this).getmConnBleDevice().getStartPosition();
        int stopPosition = BleManager.getInstance(this).getmConnBleDevice().getStopPosition();
        int position = BleManager.getInstance(this).getmConnBleDevice().getPosition();
        int smoothness = BleManager.getInstance(this).getmConnBleDevice().getSmoothness();
        int reversal = BleManager.getInstance(this).getmConnBleDevice().getReversal();
        Object obj = SharedPreferencesUtil.get(this, Constans.RemoteShareMode, true);
        send(this.mGson.toJson(new SocketSendModel(str, "", "getRemoteUrl", this.mWidth, this.mHeight, speed, mode, vibration, vibrationMode, runMode, startPosition, stopPosition, position, smoothness, deviceCode, deviceName, obj != null ? ((Boolean) obj).booleanValue() : true ? 1 : 0, reversal)));
    }

    public void heartbeat() {
        HashMap hashMap = new HashMap();
        hashMap.put(NativeProtocol.WEB_DIALOG_ACTION, "heartbeat");
        send(this.mGson.toJson(hashMap));
    }

    @Override // android.app.Service
    public IBinder onBind(Intent intent) {
        RemotePlayBinder remotePlayBinder = this.mBinder;
        if (remotePlayBinder != null) {
            return remotePlayBinder;
        }
        return null;
    }

    @Override // android.app.Service
    public void onCreate() {
        LogUtil.e("服务onCreate");
        super.onCreate();
        createNotificationChannel();
        this.mGson = new Gson();
        this.mBinder = new RemotePlayBinder();
        registerReceiver();
        connSocket(1);
    }

    @Override // android.app.Service
    public void onDestroy() {
        super.onDestroy();
        LogUtil.e("服务onDestroy");
        unregisterReceiver();
        SocketManager.getInstance().close();
        SocketManager.getInstance().closeSocket();
    }

    @Override // android.app.Service
    public int onStartCommand(Intent intent, int i, int i2) {
        LogUtil.e("服务onStartCommand");
        if (intent != null) {
            this.mWidth = intent.getIntExtra(ViewHierarchyConstants.DIMENSION_WIDTH_KEY, 0);
            this.mHeight = intent.getIntExtra(ViewHierarchyConstants.DIMENSION_HEIGHT_KEY, 0);
        }
        if (this.isError) {
            connSocket(3);
        }
        Intent intent2 = new Intent(this, (Class<?>) RemotePlayActivity.class);
        startForeground(1, new NotificationCompat.Builder(this, CHANNEL_ID).setContentTitle("My Foreground Service").setContentText("Running in the foreground").setSmallIcon(R.mipmap.ic_128_128).setContentIntent(Build.VERSION.SDK_INT >= 31 ? PendingIntent.getActivity(getApplicationContext(), 0, intent2, 67108864) : PendingIntent.getActivity(getApplicationContext(), 0, intent2, C.BUFFER_FLAG_FIRST_SAMPLE)).build());
        return 1;
    }

    @Override // android.app.Service
    public boolean onUnbind(Intent intent) {
        return super.onUnbind(intent);
    }

    public void send(String str) {
        WebSocket webSocket = mWebSocket;
        if (webSocket != null) {
            LogUtil.e("WebSocket " + mWebSocket + " 发送：" + str + ", flag:" + webSocket.send(str));
            if (this.mHandler.hasMessages(6)) {
                this.mHandler.removeMessages(6);
            }
            this.mHandler.sendEmptyMessageDelayed(6, 40000L);
        }
    }
}
