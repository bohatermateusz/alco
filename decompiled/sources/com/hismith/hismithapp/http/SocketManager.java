package com.hismith.hismithapp.http;

import android.content.Context;
import android.content.Intent;
import android.graphics.Point;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.text.TextUtils;
import com.facebook.appevents.internal.ViewHierarchyConstants;
import com.facebook.internal.NativeProtocol;
import com.facebook.share.internal.ShareConstants;
import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.hismith.hismithapp.LineRoomActivity;
import com.hismith.hismithapp.LoginActivity;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.bluetooth.BleManager;
import com.hismith.hismithapp.db.DBMessageUtil;
import com.hismith.hismithapp.model.BleDeviceModel;
import com.hismith.hismithapp.model.LineRoomInfoModel;
import com.hismith.hismithapp.model.socket.PlayLineSocketModel;
import com.hismith.hismithapp.product.ProductUtil;
import com.hismith.hismithapp.service.HuDongPlayService;
import com.hismith.hismithapp.utils.AppManager;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.MeasureUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import com.hismith.hismithapp.utils.ToastUtil;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.net.ConnectException;
import java.net.SocketTimeoutException;
import java.net.UnknownHostException;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.concurrent.TimeUnit;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;
import okhttp3.WebSocket;
import okhttp3.WebSocketListener;
import okio.ByteString;

/* loaded from: classes3.dex */
public class SocketManager {
    public static final String EXTRA_SOCKET_DATA = "socket_data";
    public static final String EXTRA_SOCKET_ERROR = "socket_error";
    public static final String EXTRA_SOCKET_TIME = "socket_time";
    public static final String RECEIVER_SOCKET_NOTIFY = "com.hismith.hismithapp.action.receiver_socket_notify";
    public static final int SOCKET_CONNECTED = 1;
    public static final int SOCKET_DISCONNECTED = 2;
    public static final int SOCKET_ERROR = 3;
    public static final int SOCKET_HEARTBEAT = 6;
    public static final int SOCKET_RECONNECT = 5;
    public static final int SOCKET_RESPONSE = 4;
    public static final int TYPE_CONN_HUDONG = 4;
    public static final int TYPE_CONN_HUDONG_SPONSOR = 5;
    public static final int TYPE_CONN_LINE = 2;
    public static final int TYPE_CONN_REMOTE = 1;
    public static final int TYPE_CONN_REREMOTE = 3;
    private static SocketManager instance = null;
    private static final int mHeartBeatDelay = 40000;
    private static WebSocket mWebSocket = null;
    public static final String welcomeMsgFlag = "*#*";
    private String clientConnection;
    private String gameToken;
    private OkHttpClient httpClient;
    private Context mContext;
    private volatile OnSocketNotifyListener onSocketNotifyListener;
    private String remoteUrl;
    private LineRoomInfoModel roomInfoModel;
    private PlayLineSocketModel roomMessageModel;
    private int mWidth = 0;
    private int mHeight = 0;
    private int connType = 0;
    private int roomId = 0;
    private int roomIsPublic = 1;
    private boolean isFailure = false;
    private boolean isConnect = false;
    private int timeout = 90000;
    private int msgId = 0;
    private long remote_time = 0;
    private Handler mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.http.SocketManager.1
        @Override // android.os.Handler.Callback
        public boolean handleMessage(Message message) {
            int i = message.what;
            if (i == 3) {
                ToastUtil.showShortToastCenter(SocketManager.this.mContext, LanguageUtil.getLanguageByKey(SocketManager.this.mContext, "networkConnectionTimedOut"));
                return false;
            }
            if (i == 4) {
                SocketManager.this.receResponse((String) message.obj);
                return false;
            }
            if (i != 6) {
                return false;
            }
            SocketManager.this.heartbeat();
            SocketManager.this.mHandler.sendEmptyMessageDelayed(6, 40000L);
            return false;
        }
    });
    private Gson mGson = new Gson();
    private RetrofitRequest baseRequest = new RetrofitRequest();

    public interface OnSocketNotifyListener {
        void OnNotify(String str);

        void onClosed(int i, String str);

        void onCodeNull();

        void onFailure(Throwable th, Response response);

        void onOpen();
    }

    public SocketManager(Context context) {
        this.mContext = context;
    }

    private WebSocketListener createListener() {
        return new WebSocketListener() { // from class: com.hismith.hismithapp.http.SocketManager.2
            @Override // okhttp3.WebSocketListener
            public void onClosed(WebSocket webSocket, int i, String str) {
                super.onClosed(webSocket, i, str);
                LogUtil.e("WebSocket: onClosed code:" + i + ", reason:" + str);
                if (webSocket == SocketManager.mWebSocket || SocketManager.mWebSocket == null) {
                    WebSocket unused = SocketManager.mWebSocket = null;
                    if (SocketManager.this.msgId != 0) {
                        SocketManager.this.sendNotifyTime();
                    }
                    SocketManager.this.remoteUrl = "";
                    SocketManager.this.clientConnection = "";
                    SocketManager.this.isConnect = false;
                    SocketManager.this.isFailure = true;
                    SocketManager.this.sendNotify(1, "onClosed");
                    SocketManager.this.clearHandler();
                    if (SocketManager.this.onSocketNotifyListener != null) {
                        SocketManager.this.onSocketNotifyListener.onClosed(i, str);
                    }
                }
            }

            @Override // okhttp3.WebSocketListener
            public void onClosing(WebSocket webSocket, int i, String str) {
                super.onClosing(webSocket, i, str);
                LogUtil.e("onClosing code:" + i + ", reason:" + str);
                webSocket.close(i, str);
            }

            @Override // okhttp3.WebSocketListener
            public void onFailure(WebSocket webSocket, Throwable th, Response response) {
                super.onFailure(webSocket, th, response);
                StringWriter stringWriter = new StringWriter();
                PrintWriter printWriter = new PrintWriter(stringWriter);
                th.printStackTrace(printWriter);
                for (Throwable cause = th.getCause(); cause != null; cause = cause.getCause()) {
                    cause.printStackTrace(printWriter);
                }
                printWriter.close();
                LogUtil.e("WebSocket: onFailure：" + stringWriter.toString() + ", webSocket:" + webSocket + ", mWebSocket:" + SocketManager.mWebSocket + ", 是否和当前webSocket相等" + (webSocket == SocketManager.mWebSocket));
                if (webSocket == SocketManager.mWebSocket || SocketManager.mWebSocket == null) {
                    WebSocket unused = SocketManager.mWebSocket = null;
                    if (SocketManager.this.msgId != 0) {
                        SocketManager.this.sendNotifyTime();
                    }
                    SocketManager.this.remoteUrl = "";
                    SocketManager.this.clientConnection = "";
                    SocketManager.this.isConnect = false;
                    SocketManager.this.isFailure = true;
                    SocketManager.this.sendNotify(1, "onFailure");
                    SocketManager.this.clearHandler();
                    if ((th instanceof ConnectException) || (th instanceof SocketTimeoutException) || (th instanceof UnknownHostException)) {
                        SocketManager.this.mHandler.sendEmptyMessage(3);
                    }
                    LogUtil.e("WebSocket: onFailure：" + th.getClass());
                    if (SocketManager.this.onSocketNotifyListener != null) {
                        SocketManager.this.onSocketNotifyListener.onFailure(th, response);
                    }
                }
            }

            @Override // okhttp3.WebSocketListener
            public void onMessage(WebSocket webSocket, String str) {
                super.onMessage(webSocket, str);
                if (SocketManager.mWebSocket == null) {
                    WebSocket unused = SocketManager.mWebSocket = webSocket;
                }
                SocketManager.this.isConnect = true;
                SocketManager.this.isFailure = false;
                Message message = new Message();
                message.what = 4;
                message.obj = str;
                SocketManager.this.mHandler.sendMessage(message);
            }

            @Override // okhttp3.WebSocketListener
            public void onMessage(WebSocket webSocket, ByteString byteString) {
                super.onMessage(webSocket, byteString);
                SocketManager.this.isConnect = true;
            }

            @Override // okhttp3.WebSocketListener
            public void onOpen(WebSocket webSocket, Response response) {
                super.onOpen(webSocket, response);
                SocketManager.this.isConnect = true;
                SocketManager.this.close();
                SocketManager.this.closeSocket();
                LogUtil.e("WebSocket: onOpen:" + webSocket + ", type:" + SocketManager.this.connType);
                WebSocket unused = SocketManager.mWebSocket = webSocket;
                int i = SocketManager.this.connType;
                SocketManager socketManager = SocketManager.this;
                if (i == 1) {
                    socketManager.getRemoteUrl();
                } else {
                    int i2 = socketManager.connType;
                    SocketManager socketManager2 = SocketManager.this;
                    if (i2 == 2) {
                        HuDongPlayService.stopService(socketManager2.mContext);
                        SocketManager.this.onLinePlay();
                        SocketManager.this.worldUsers(true);
                        SocketManager.this.worldRooms(true);
                    } else {
                        int i3 = socketManager2.connType;
                        SocketManager socketManager3 = SocketManager.this;
                        if (i3 == 3) {
                            socketManager3.getReRemote();
                        } else {
                            int i4 = socketManager3.connType;
                            SocketManager socketManager4 = SocketManager.this;
                            if (i4 == 4) {
                                socketManager4.hudongConnection(socketManager4.gameToken, null);
                            } else if (socketManager4.connType == 5) {
                                HuDongPlayService.stopService(SocketManager.this.mContext);
                                SocketManager.this.getRemoteUrl();
                            }
                        }
                    }
                }
                if (SocketManager.this.onSocketNotifyListener != null) {
                    SocketManager.this.onSocketNotifyListener.onOpen();
                }
            }
        };
    }

    public static SocketManager getInstance() {
        SocketManager socketManager;
        synchronized (SocketManager.class) {
            socketManager = instance;
            if (socketManager == null) {
                socketManager = new SocketManager(MyApplication.getApplication());
                instance = socketManager;
            }
        }
        return socketManager;
    }

    private void initScreenSize() {
        if (this.mWidth == 0 || this.mHeight == 0) {
            Point screenSize = MeasureUtil.getScreenSize(this.mContext);
            this.mWidth = screenSize.x;
            this.mHeight = screenSize.y;
        }
    }

    private String modifyDatField(String str, String str2) {
        Matcher matcher = Pattern.compile("(\"" + str + "\"\\s*:\\s*\")([^\"]*)(\")").matcher(str2);
        if (!matcher.find()) {
            return str2;
        }
        return str2.replace(matcher.group(0), matcher.group(1) + "长度" + matcher.group(2).length() + "个字符" + matcher.group(3));
    }

    /* JADX INFO: Access modifiers changed from: private */
    /* JADX WARN: Removed duplicated region for block: B:39:0x02e6 A[Catch: Exception -> 0x02ec, TRY_LEAVE, TryCatch #0 {Exception -> 0x02ec, blocks: (B:5:0x005c, B:10:0x007e, B:13:0x008f, B:16:0x00a1, B:18:0x00a7, B:20:0x00ad, B:21:0x00b2, B:23:0x00ba, B:25:0x00ce, B:26:0x00d6, B:34:0x0192, B:37:0x02de, B:39:0x02e6, B:44:0x0098, B:45:0x0087, B:48:0x019a, B:51:0x01ab, B:53:0x01b1, B:54:0x01bb, B:56:0x01c3, B:57:0x01cf, B:59:0x01d7, B:60:0x01e3, B:62:0x01eb, B:63:0x01f1, B:65:0x01f9, B:69:0x020d, B:70:0x0203, B:71:0x0216, B:74:0x0220, B:77:0x022f, B:78:0x0227, B:79:0x0233, B:81:0x023b, B:84:0x024d, B:87:0x025e, B:91:0x0275, B:94:0x0287, B:95:0x027e, B:96:0x026c, B:97:0x0256, B:98:0x0245, B:99:0x029e, B:101:0x02a6, B:102:0x02b5, B:104:0x02bf, B:106:0x02c7, B:108:0x02cf, B:109:0x01a3, B:110:0x0072, B:28:0x0173, B:30:0x0185, B:31:0x0188), top: B:4:0x005c, inners: #1 }] */
    /* JADX WARN: Removed duplicated region for block: B:43:? A[RETURN, SYNTHETIC] */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public void receResponse(String str) {
        String str2;
        if (TextUtils.isEmpty(str)) {
            return;
        }
        LogUtil.e("WebSocket " + mWebSocket + " : 接收:" + modifyDatField("dat", str));
        try {
            JsonObject asJsonObject = new JsonParser().parse(str).getAsJsonObject();
            int asInt = asJsonObject.get("success") == null ? 0 : asJsonObject.get("success").getAsInt();
            if (asInt == 0) {
                String asString = asJsonObject.get("errorCode") == null ? "" : asJsonObject.get("errorCode").getAsString();
                String asString2 = asJsonObject.get("errorMessage") != null ? asJsonObject.get("errorMessage").getAsString() : "";
                if (!TextUtils.isEmpty(asString)) {
                    if (!TextUtils.isEmpty(asString2)) {
                        ToastUtil.showShortToastCenter(this.mContext, asString2);
                    }
                    if (Constans.errorRelogin.equals(asString)) {
                        String str3 = (String) SharedPreferencesUtil.get(this.mContext, Constans.Language);
                        Object obj = SharedPreferencesUtil.get(this.mContext, Constans.contactUserId, 0);
                        int intValue = obj != null ? ((Integer) obj).intValue() : 0;
                        String str4 = (String) SharedPreferencesUtil.get(this.mContext, Constans.storeUrl);
                        String str5 = (String) SharedPreferencesUtil.get(this.mContext, Constans.roomVoice);
                        String str6 = (String) SharedPreferencesUtil.get(this.mContext, Constans.tempToken);
                        String str7 = (String) SharedPreferencesUtil.get(this.mContext, Constans.aboutApp);
                        String str8 = (String) SharedPreferencesUtil.get(this.mContext, Constans.copyright);
                        int i = intValue;
                        String str9 = (String) SharedPreferencesUtil.get(this.mContext, Constans.mainBackgroundUrl);
                        String str10 = (String) SharedPreferencesUtil.get(this.mContext, Constans.bluetoothConnAddress);
                        String str11 = (String) SharedPreferencesUtil.get(this.mContext, Constans.ProductAllData);
                        MyApplication.getApplication().setmUserModel(null);
                        SharedPreferencesUtil.clear(this.mContext);
                        SharedPreferencesUtil.put(this.mContext, Constans.Language, str3);
                        SharedPreferencesUtil.put(this.mContext, Constans.contactUserId, Integer.valueOf(i));
                        SharedPreferencesUtil.put(this.mContext, Constans.storeUrl, str4);
                        SharedPreferencesUtil.put(this.mContext, Constans.roomVoice, str5);
                        SharedPreferencesUtil.put(this.mContext, Constans.tempToken, str6);
                        SharedPreferencesUtil.put(this.mContext, Constans.aboutApp, str7);
                        SharedPreferencesUtil.put(this.mContext, Constans.copyright, str8);
                        SharedPreferencesUtil.put(this.mContext, Constans.mainBackgroundUrl, str9);
                        SharedPreferencesUtil.put(this.mContext, Constans.bluetoothConnAddress, str10);
                        SharedPreferencesUtil.put(this.mContext, Constans.ProductAllData, str11);
                        try {
                            Intent intent = new Intent(this.mContext, (Class<?>) LoginActivity.class);
                            intent.addFlags(268435456);
                            Context context = this.mContext;
                            if (context != null) {
                                context.startActivity(intent);
                            }
                            AppManager.getInstance().finishAllOtherActivity(LoginActivity.class);
                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                    }
                }
            } else {
                int i2 = 1;
                if (1 == asInt) {
                    String asString3 = asJsonObject.get(NativeProtocol.WEB_DIALOG_ACTION) == null ? "" : asJsonObject.get(NativeProtocol.WEB_DIALOG_ACTION).getAsString();
                    if (asJsonObject.get("gameToken") != null) {
                        this.gameToken = asJsonObject.get("gameToken").getAsString();
                    }
                    if (asJsonObject.get("remoteUrl") != null) {
                        this.remoteUrl = asJsonObject.get("remoteUrl").getAsString();
                    } else if (asJsonObject.get("clientConnection") != null) {
                        this.clientConnection = asJsonObject.get("clientConnection").getAsString();
                    } else if (asJsonObject.get("offline") != null) {
                        this.clientConnection = "0";
                    } else if ("onlinePlay".equals(asString3)) {
                        int asInt2 = asJsonObject.get("userRoomId") == null ? 0 : asJsonObject.get("userRoomId").getAsInt();
                        if (asInt2 > 0) {
                            getInstance().enterRoom(asInt2);
                        }
                    } else if ("createRoom".equals(asString3)) {
                        if (asJsonObject.get("isPublic") != null) {
                            i2 = asJsonObject.get("isPublic").getAsInt();
                        }
                        this.roomIsPublic = i2;
                    } else {
                        if (!"enterRoom".equals(asString3)) {
                            if ("roomInfo".equals(asString3)) {
                                str2 = str;
                                this.roomInfoModel = (LineRoomInfoModel) this.mGson.fromJson(str2, LineRoomInfoModel.class);
                            } else {
                                str2 = str;
                                if ("exitRoom".equals(asString3) || "closeRoom".equals(asString3) || "removeRoom".equals(asString3)) {
                                    this.roomInfoModel = null;
                                    AppManager.getInstance().finishActivity(LineRoomActivity.class);
                                }
                            }
                            sendNotify(0, str2);
                            if (this.onSocketNotifyListener == null) {
                                this.onSocketNotifyListener.OnNotify(str2);
                                return;
                            }
                            return;
                        }
                        this.roomId = asJsonObject.get("roomId") == null ? 0 : asJsonObject.get("roomId").getAsInt();
                        if (asJsonObject.get("isPublic") != null) {
                            i2 = asJsonObject.get("isPublic").getAsInt();
                        }
                        this.roomIsPublic = i2;
                        int asInt3 = asJsonObject.get("userId") == null ? 0 : asJsonObject.get("userId").getAsInt();
                        String asString4 = asJsonObject.get("welcomeMsg") != null ? asJsonObject.get("welcomeMsg").getAsString() : "";
                        PlayLineSocketModel playLineSocketModel = new PlayLineSocketModel();
                        this.roomMessageModel = playLineSocketModel;
                        playLineSocketModel.setMessage(asString4);
                        this.roomMessageModel.setUserId(asInt3);
                        this.roomMessageModel.setUserName("*#*");
                    }
                }
            }
            str2 = str;
            sendNotify(0, str2);
            if (this.onSocketNotifyListener == null) {
            }
        } catch (Exception e2) {
            e2.printStackTrace();
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void sendNotify(int i, String str) {
        if (i != 0) {
            LogUtil.e("SocketManager Socket广播 text:" + str);
        }
        Intent intent = new Intent(RECEIVER_SOCKET_NOTIFY);
        Bundle bundle = new Bundle();
        bundle.putInt(EXTRA_SOCKET_ERROR, i);
        bundle.putString(EXTRA_SOCKET_DATA, str);
        intent.putExtras(bundle);
        this.mContext.sendBroadcast(intent);
    }

    public void agreeUserRoom(String str) {
        HashMap hashMap = new HashMap();
        hashMap.put(NativeProtocol.WEB_DIALOG_ACTION, "agreeUserRoom");
        hashMap.put("userId", str);
        send(this.mGson.toJson(hashMap));
    }

    public void chatMessage(String str) {
        HashMap hashMap = new HashMap();
        hashMap.put(NativeProtocol.WEB_DIALOG_ACTION, "chatMessage");
        hashMap.put(ShareConstants.WEB_DIALOG_PARAM_MESSAGE, str);
        send(this.mGson.toJson(hashMap));
    }

    public void clearHandler() {
        this.mHandler.removeCallbacksAndMessages(null);
    }

    public void close() {
        HashMap hashMap = new HashMap();
        hashMap.put(NativeProtocol.WEB_DIALOG_ACTION, "close");
        send(this.mGson.toJson(hashMap));
    }

    public void closeSocket() {
        LogUtil.e("断开Socket mWebSocket:" + mWebSocket);
        this.remoteUrl = "";
        this.clientConnection = "";
        WebSocket webSocket = mWebSocket;
        if (webSocket != null) {
            webSocket.close(1000, "");
            mWebSocket = null;
        }
    }

    public void connSocket(int i) {
        this.connType = i;
        this.httpClient = new OkHttpClient.Builder().connectTimeout(0L, TimeUnit.MILLISECONDS).pingInterval(40L, TimeUnit.SECONDS).dns(new XDns(this.timeout)).build();
        String str = this.connType == 2 ? "ws://www.hismiths.com:8887" : "ws://www.hismiths.com:8886";
        LogUtil.e("connSocket:".concat(str));
        this.httpClient.newWebSocket(new Request.Builder().url(str).build(), createListener());
        this.isFailure = false;
        this.isConnect = false;
        LogUtil.e("WebSocket 连接Socket");
    }

    public void createRoom(String str, int i, int i2) {
        HashMap hashMap = new HashMap();
        hashMap.put(NativeProtocol.WEB_DIALOG_ACTION, "createRoom");
        hashMap.put("roomName", str);
        hashMap.put("userLimit", Integer.valueOf(i));
        hashMap.put("isPublic", Integer.valueOf(i2));
        send(this.mGson.toJson(hashMap));
    }

    public void enterRoom(int i) {
        HashMap hashMap = new HashMap();
        hashMap.put(NativeProtocol.WEB_DIALOG_ACTION, "enterRoom");
        hashMap.put("roomId", Integer.valueOf(i));
        send(this.mGson.toJson(hashMap));
    }

    public void exitRoom() {
        HashMap hashMap = new HashMap();
        hashMap.put(NativeProtocol.WEB_DIALOG_ACTION, "exitRoom");
        send(this.mGson.toJson(hashMap));
    }

    public String getClientConnection() {
        return this.clientConnection;
    }

    public String getGameToken() {
        return this.gameToken;
    }

    public int getMsgId() {
        return this.msgId;
    }

    public OnSocketNotifyListener getOnSocketNotifyListener() {
        return this.onSocketNotifyListener;
    }

    /* JADX WARN: Multi-variable type inference failed */
    public void getReRemote() {
        initScreenSize();
        String str = (String) SharedPreferencesUtil.get(this.mContext, Constans.Token);
        String deviceCode = MyApplication.getApplication().getDeviceCode();
        if (TextUtils.isEmpty(deviceCode)) {
            if (this.onSocketNotifyListener != null) {
                this.onSocketNotifyListener.onCodeNull();
                return;
            }
            return;
        }
        String deviceName = MyApplication.getApplication().getDeviceName();
        LinkedHashMap linkedHashMap = new LinkedHashMap();
        linkedHashMap.put(Constans.Token, str);
        linkedHashMap.put("appBrand", "");
        linkedHashMap.put("gameToken", this.gameToken);
        linkedHashMap.put(NativeProtocol.WEB_DIALOG_ACTION, "getRemoteUrl");
        linkedHashMap.put("productId", deviceCode);
        linkedHashMap.put("productName", deviceName);
        linkedHashMap.put(ViewHierarchyConstants.DIMENSION_WIDTH_KEY, Integer.valueOf(this.mWidth));
        linkedHashMap.put(ViewHierarchyConstants.DIMENSION_HEIGHT_KEY, Integer.valueOf(this.mHeight));
        linkedHashMap.put("version", "V2");
        Object obj = SharedPreferencesUtil.get(this.mContext, Constans.RemoteShareMode, true);
        linkedHashMap.put("modeStatus", Integer.valueOf(obj != null ? ((Boolean) obj).booleanValue() : 1));
        if (BleManager.getInstance(this.mContext).getmConnBleDevice() != null) {
            int speed = BleManager.getInstance(this.mContext).getmConnBleDevice().getSpeed();
            int mode = BleManager.getInstance(this.mContext).getmConnBleDevice().getMode();
            linkedHashMap.put("speed", Integer.valueOf(speed));
            linkedHashMap.put("mode", Integer.valueOf(mode));
            if (ProductUtil.isSMVType()) {
                int vibration = BleManager.getInstance(this.mContext).getmConnBleDevice().getVibration();
                int vibrationMode = BleManager.getInstance(this.mContext).getmConnBleDevice().getVibrationMode();
                linkedHashMap.put("vspeed", Integer.valueOf(vibration));
                linkedHashMap.put("vibration", Integer.valueOf(vibrationMode));
            } else if (ProductUtil.isSM2Type()) {
                linkedHashMap.put("vibration", Integer.valueOf(BleManager.getInstance(this.mContext).getmConnBleDevice().getVibrationMode()));
            } else if (ProductUtil.isSMSType()) {
                int runMode = BleManager.getInstance(this.mContext).getmConnBleDevice().getRunMode();
                int startPosition = BleManager.getInstance(this.mContext).getmConnBleDevice().getStartPosition();
                int stopPosition = BleManager.getInstance(this.mContext).getmConnBleDevice().getStopPosition();
                int position = BleManager.getInstance(this.mContext).getmConnBleDevice().getPosition();
                int smoothness = BleManager.getInstance(this.mContext).getmConnBleDevice().getSmoothness();
                linkedHashMap.put("strokeType", Integer.valueOf(runMode));
                linkedHashMap.put("strokeMin", Integer.valueOf(startPosition));
                linkedHashMap.put("strokeMax", Integer.valueOf(stopPosition));
                linkedHashMap.put("strokeNow", Integer.valueOf(position));
                linkedHashMap.put("smooth", Integer.valueOf(smoothness));
            }
            if (ProductUtil.hasReversal()) {
                linkedHashMap.put("reversal", Integer.valueOf(BleManager.getInstance(this.mContext).getmConnBleDevice().getReversal()));
            }
            if (BleManager.getInstance(this.mContext).hasPiuFunBle()) {
                linkedHashMap.put("inject", 1);
            }
        }
        send(this.mGson.toJson(linkedHashMap));
    }

    /* JADX WARN: Multi-variable type inference failed */
    public void getRemoteUrl() {
        String str = (String) SharedPreferencesUtil.get(this.mContext, Constans.Token);
        String deviceCode = BleManager.getInstance(this.mContext).getmConnBleDevice().getDeviceCode();
        if (TextUtils.isEmpty(deviceCode)) {
            if (this.onSocketNotifyListener != null) {
                this.onSocketNotifyListener.onCodeNull();
                return;
            }
            return;
        }
        String deviceName = BleManager.getInstance(this.mContext).getmConnBleDevice().getDeviceName();
        initScreenSize();
        LinkedHashMap linkedHashMap = new LinkedHashMap();
        linkedHashMap.put(Constans.Token, str);
        linkedHashMap.put("appBrand", "");
        linkedHashMap.put(NativeProtocol.WEB_DIALOG_ACTION, "getRemoteUrl");
        linkedHashMap.put(ViewHierarchyConstants.DIMENSION_WIDTH_KEY, Integer.valueOf(this.mWidth));
        linkedHashMap.put(ViewHierarchyConstants.DIMENSION_HEIGHT_KEY, Integer.valueOf(this.mHeight));
        linkedHashMap.put("version", "V2");
        linkedHashMap.put("productId", deviceCode);
        linkedHashMap.put("productName", deviceName);
        int speed = BleManager.getInstance(this.mContext).getmConnBleDevice().getSpeed();
        int mode = BleManager.getInstance(this.mContext).getmConnBleDevice().getMode();
        linkedHashMap.put("speed", Integer.valueOf(speed));
        linkedHashMap.put("mode", Integer.valueOf(mode));
        if (ProductUtil.isSMVType()) {
            int vibration = BleManager.getInstance(this.mContext).getmConnBleDevice().getVibration();
            int vibrationMode = BleManager.getInstance(this.mContext).getmConnBleDevice().getVibrationMode();
            linkedHashMap.put("vspeed", Integer.valueOf(vibration));
            linkedHashMap.put("vibration", Integer.valueOf(vibrationMode));
        } else if (ProductUtil.isSM2Type()) {
            linkedHashMap.put("vibration", Integer.valueOf(BleManager.getInstance(this.mContext).getmConnBleDevice().getVibrationMode()));
        } else if (ProductUtil.isSMSType()) {
            int runMode = BleManager.getInstance(this.mContext).getmConnBleDevice().getRunMode();
            int startPosition = BleManager.getInstance(this.mContext).getmConnBleDevice().getStartPosition();
            int stopPosition = BleManager.getInstance(this.mContext).getmConnBleDevice().getStopPosition();
            int position = BleManager.getInstance(this.mContext).getmConnBleDevice().getPosition();
            int smoothness = BleManager.getInstance(this.mContext).getmConnBleDevice().getSmoothness();
            linkedHashMap.put("strokeType", Integer.valueOf(runMode));
            linkedHashMap.put("strokeMin", Integer.valueOf(startPosition));
            linkedHashMap.put("strokeMax", Integer.valueOf(stopPosition));
            linkedHashMap.put("strokeNow", Integer.valueOf(position));
            linkedHashMap.put("smooth", Integer.valueOf(smoothness));
        }
        if (ProductUtil.hasReversal()) {
            linkedHashMap.put("reversal", Integer.valueOf(BleManager.getInstance(this.mContext).getmConnBleDevice().getReversal()));
        }
        if (BleManager.getInstance(this.mContext).hasPiuFunBle()) {
            linkedHashMap.put("inject", 1);
        }
        Object obj = SharedPreferencesUtil.get(this.mContext, Constans.RemoteShareMode, true);
        linkedHashMap.put("modeStatus", Integer.valueOf(obj != null ? ((Boolean) obj).booleanValue() : 1));
        send(this.mGson.toJson(linkedHashMap));
    }

    public long getRemote_time() {
        return this.remote_time;
    }

    public int getRoomId() {
        return this.roomId;
    }

    public LineRoomInfoModel getRoomInfoModel() {
        return this.roomInfoModel;
    }

    public int getRoomIsPublic() {
        return this.roomIsPublic;
    }

    public PlayLineSocketModel getRoomMessageModel() {
        return this.roomMessageModel;
    }

    public void getRooms(int i) {
        String deviceCode = MyApplication.getApplication().getDeviceCode();
        if (TextUtils.isEmpty(deviceCode)) {
            if (this.onSocketNotifyListener != null) {
                this.onSocketNotifyListener.onCodeNull();
            }
        } else {
            HashMap hashMap = new HashMap();
            hashMap.put(NativeProtocol.WEB_DIALOG_ACTION, "getRooms");
            hashMap.put("productId", deviceCode);
            hashMap.put("new", Integer.valueOf(i));
            send(this.mGson.toJson(hashMap));
        }
    }

    public WebSocket getWebSocket() {
        return mWebSocket;
    }

    public void hallUser() {
        HashMap hashMap = new HashMap();
        hashMap.put(NativeProtocol.WEB_DIALOG_ACTION, "hallUser");
        send(this.mGson.toJson(hashMap));
    }

    public void heartbeat() {
        HashMap hashMap = new HashMap();
        hashMap.put(NativeProtocol.WEB_DIALOG_ACTION, "heartbeat");
        send(this.mGson.toJson(hashMap));
    }

    /* JADX WARN: Multi-variable type inference failed */
    public void hudongConnection(String str, BleDeviceModel bleDeviceModel) {
        initScreenSize();
        String str2 = (String) SharedPreferencesUtil.get(this.mContext, Constans.Token);
        LinkedHashMap linkedHashMap = new LinkedHashMap();
        linkedHashMap.put(Constans.Token, str2);
        linkedHashMap.put("gameToken", str);
        linkedHashMap.put(NativeProtocol.WEB_DIALOG_ACTION, "connection");
        linkedHashMap.put(ViewHierarchyConstants.DIMENSION_WIDTH_KEY, Integer.valueOf(this.mWidth));
        linkedHashMap.put(ViewHierarchyConstants.DIMENSION_HEIGHT_KEY, Integer.valueOf(this.mHeight));
        linkedHashMap.put("appBrand", "");
        linkedHashMap.put("version", "V2");
        Object obj = SharedPreferencesUtil.get(this.mContext, Constans.RemoteShareMode, true);
        linkedHashMap.put("modeStatus", Integer.valueOf(obj != null ? ((Boolean) obj).booleanValue() : 1));
        if (bleDeviceModel != null) {
            linkedHashMap.put("productId", bleDeviceModel.getDeviceCode());
            linkedHashMap.put("productName", bleDeviceModel.getDeviceName());
            linkedHashMap.put("speed", Integer.valueOf(bleDeviceModel.getSpeed()));
            linkedHashMap.put("mode", Integer.valueOf(bleDeviceModel.getMode()));
            if (ProductUtil.isSMVType()) {
                linkedHashMap.put("vspeed", Integer.valueOf(bleDeviceModel.getVibration()));
                linkedHashMap.put("vibration", Integer.valueOf(bleDeviceModel.getVibrationMode()));
            } else if (ProductUtil.isSM2Type()) {
                linkedHashMap.put("vibration", Integer.valueOf(bleDeviceModel.getVibrationMode()));
            } else if (ProductUtil.isSMSType()) {
                linkedHashMap.put("strokeType", Integer.valueOf(bleDeviceModel.getRunMode()));
                linkedHashMap.put("strokeMin", Integer.valueOf(bleDeviceModel.getStartPosition()));
                linkedHashMap.put("strokeMax", Integer.valueOf(bleDeviceModel.getStopPosition()));
                linkedHashMap.put("strokeNow", Integer.valueOf(bleDeviceModel.getPosition()));
                linkedHashMap.put("smooth", Integer.valueOf(bleDeviceModel.getSmoothness()));
            }
        }
        send(this.mGson.toJson(linkedHashMap));
    }

    public void inviteUserRoom(int i) {
        HashMap hashMap = new HashMap();
        hashMap.put(NativeProtocol.WEB_DIALOG_ACTION, "inviteUserRoom");
        hashMap.put("userId", Integer.valueOf(i));
        send(this.mGson.toJson(hashMap));
    }

    public boolean isConnect() {
        return this.isConnect;
    }

    public boolean isHuDongConnect() {
        if (mWebSocket == null || this.isFailure) {
            return false;
        }
        int i = this.connType;
        return i == 4 || i == 5;
    }

    public boolean isRemoteConnect() {
        return (mWebSocket == null || this.isFailure || this.connType != 1 || TextUtils.isEmpty(this.remoteUrl)) ? false : true;
    }

    public void onLinePlay() {
        initScreenSize();
        String str = (String) SharedPreferencesUtil.get(this.mContext, Constans.Token);
        String deviceCode = MyApplication.getApplication().getDeviceCode();
        if (TextUtils.isEmpty(deviceCode)) {
            if (this.onSocketNotifyListener != null) {
                this.onSocketNotifyListener.onCodeNull();
                return;
            }
            return;
        }
        HashMap hashMap = new HashMap();
        hashMap.put(Constans.Token, str);
        hashMap.put("appBrand", "");
        hashMap.put(NativeProtocol.WEB_DIALOG_ACTION, "onlinePlay");
        hashMap.put("productId", deviceCode);
        hashMap.put(ViewHierarchyConstants.DIMENSION_WIDTH_KEY, Integer.valueOf(this.mWidth));
        hashMap.put(ViewHierarchyConstants.DIMENSION_HEIGHT_KEY, Integer.valueOf(this.mHeight));
        send(this.mGson.toJson(hashMap));
    }

    public void removeUserRoom(int i) {
        HashMap hashMap = new HashMap();
        hashMap.put(NativeProtocol.WEB_DIALOG_ACTION, "removeUserRoom");
        hashMap.put("userId", Integer.valueOf(i));
        send(this.mGson.toJson(hashMap));
    }

    public void roomCaptain(int i) {
        HashMap hashMap = new HashMap();
        hashMap.put(NativeProtocol.WEB_DIALOG_ACTION, "roomCaptain");
        hashMap.put("userId", Integer.valueOf(i));
        send(this.mGson.toJson(hashMap));
    }

    public void roomGetMyModes() {
        HashMap hashMap = new HashMap();
        hashMap.put(NativeProtocol.WEB_DIALOG_ACTION, "roomGetMyModes");
        send(this.mGson.toJson(hashMap));
    }

    public void roomInfo() {
        HashMap hashMap = new HashMap();
        hashMap.put(NativeProtocol.WEB_DIALOG_ACTION, "roomInfo");
        send(this.mGson.toJson(hashMap));
    }

    public void roomMessage(String str) {
        HashMap hashMap = new HashMap();
        hashMap.put(NativeProtocol.WEB_DIALOG_ACTION, "roomMessage");
        hashMap.put(ShareConstants.WEB_DIALOG_PARAM_MESSAGE, str);
        send(this.mGson.toJson(hashMap));
    }

    public void roomSetMode(int i, int i2) {
        HashMap hashMap = new HashMap();
        hashMap.put(NativeProtocol.WEB_DIALOG_ACTION, "roomSetMode");
        hashMap.put("modeId", Integer.valueOf(i));
        hashMap.put("playTime", Integer.valueOf(i2));
        send(this.mGson.toJson(hashMap));
    }

    public void roomStopMode() {
        HashMap hashMap = new HashMap();
        hashMap.put(NativeProtocol.WEB_DIALOG_ACTION, "roomStopMode");
        send(this.mGson.toJson(hashMap));
    }

    public void roomUser() {
        HashMap hashMap = new HashMap();
        hashMap.put(NativeProtocol.WEB_DIALOG_ACTION, "roomUser");
        send(this.mGson.toJson(hashMap));
    }

    public boolean send(String str) {
        if (this.isFailure) {
            LogUtil.e("WebSocket isFailure");
            sendNotify(1, "NULL");
            if (this.onSocketNotifyListener != null) {
                this.onSocketNotifyListener.onClosed(0, "NULL");
                return false;
            }
        } else {
            WebSocket webSocket = mWebSocket;
            if (webSocket != null) {
                boolean send = webSocket.send(str);
                LogUtil.e("WebSocket " + mWebSocket + " 发送：" + str + ", flag:" + send);
                if (this.mHandler.hasMessages(6)) {
                    this.mHandler.removeMessages(6);
                }
                this.mHandler.sendEmptyMessageDelayed(6, 40000L);
                return send;
            }
        }
        return false;
    }

    public void sendNotifyTime() {
        DBMessageUtil.updateDBRemoteTime(this.mContext, this.msgId + "", (int) (System.currentTimeMillis() - this.remote_time));
        this.msgId = 0;
        this.remote_time = 0L;
    }

    public void setClimax() {
        HashMap hashMap = new HashMap();
        hashMap.put(NativeProtocol.WEB_DIALOG_ACTION, "setClimax");
        send(this.mGson.toJson(hashMap));
    }

    public void setGameToken(String str) {
        this.gameToken = str;
    }

    public void setMode(int i) {
        setMode(i, 0);
    }

    public void setMode(int i, int i2) {
        HashMap hashMap = new HashMap();
        hashMap.put(NativeProtocol.WEB_DIALOG_ACTION, "setMode");
        hashMap.put("funId", 1);
        hashMap.put("mode", Integer.valueOf(i));
        hashMap.put("onlyRead", Integer.valueOf(i2));
        send(this.mGson.toJson(hashMap));
    }

    public void setMsgId(int i) {
        this.msgId = i;
    }

    public void setOnSocketNotifyListener(OnSocketNotifyListener onSocketNotifyListener) {
        this.onSocketNotifyListener = onSocketNotifyListener;
    }

    public void setRemote_time(long j) {
        this.remote_time = j;
    }

    public void setReversal(int i) {
        setReversal(i, 0);
    }

    public void setReversal(int i, int i2) {
        HashMap hashMap = new HashMap();
        hashMap.put(NativeProtocol.WEB_DIALOG_ACTION, "setReversal");
        hashMap.put("reversal", Integer.valueOf(i));
        hashMap.put("onlyRead", Integer.valueOf(i2));
        send(this.mGson.toJson(hashMap));
    }

    public void setRoomMessageModel(PlayLineSocketModel playLineSocketModel) {
        this.roomMessageModel = playLineSocketModel;
    }

    public SocketManager setScreenSize(int i, int i2) {
        this.mWidth = i;
        this.mHeight = i2;
        return instance;
    }

    public void setSmooth(int i) {
        setSmooth(i, 0);
    }

    public void setSmooth(int i, int i2) {
        HashMap hashMap = new HashMap();
        hashMap.put(NativeProtocol.WEB_DIALOG_ACTION, "setSmooth");
        hashMap.put("smooth", Integer.valueOf(i));
        hashMap.put("onlyRead", Integer.valueOf(i2));
        send(this.mGson.toJson(hashMap));
    }

    public void setSpeed(int i) {
        setSpeed(i, 0);
    }

    public void setSpeed(int i, int i2) {
        HashMap hashMap = new HashMap();
        hashMap.put(NativeProtocol.WEB_DIALOG_ACTION, "setSpeed");
        hashMap.put("funId", 1);
        hashMap.put("speed", Integer.valueOf(i));
        hashMap.put("onlyRead", Integer.valueOf(i2));
        send(this.mGson.toJson(hashMap));
    }

    public void setSpeed(int i, JsonObject jsonObject) {
        HashMap hashMap = new HashMap();
        hashMap.put(NativeProtocol.WEB_DIALOG_ACTION, "setSpeed");
        hashMap.put("funId", 1);
        hashMap.put("speed", Integer.valueOf(i));
        hashMap.put("lines", jsonObject);
        send(this.mGson.toJson(hashMap));
    }

    public void setStroke(int i, int i2) {
        setStroke(i, i2, 0);
    }

    public void setStroke(int i, int i2, int i3) {
        HashMap hashMap = new HashMap();
        hashMap.put(NativeProtocol.WEB_DIALOG_ACTION, "setStroke");
        hashMap.put("strokeMin", Integer.valueOf(i));
        hashMap.put("strokeMax", Integer.valueOf(i2));
        hashMap.put("onlyRead", Integer.valueOf(i3));
        send(this.mGson.toJson(hashMap));
    }

    public void setStrokeNow(int i) {
        setStrokeNow(i, 0);
    }

    public void setStrokeNow(int i, int i2) {
        HashMap hashMap = new HashMap();
        hashMap.put(NativeProtocol.WEB_DIALOG_ACTION, "setStrokeNow");
        hashMap.put("strokeNow", Integer.valueOf(i));
        hashMap.put("onlyRead", Integer.valueOf(i2));
        send(this.mGson.toJson(hashMap));
    }

    public void setStrokeType(int i) {
        setStrokeType(i, 0);
    }

    public void setStrokeType(int i, int i2) {
        HashMap hashMap = new HashMap();
        hashMap.put(NativeProtocol.WEB_DIALOG_ACTION, "setStrokeType");
        hashMap.put("strokeType", Integer.valueOf(i));
        hashMap.put("onlyRead", Integer.valueOf(i2));
        send(this.mGson.toJson(hashMap));
    }

    public void setVibrate(int i) {
        setVibrate(i, 0);
    }

    public void setVibrate(int i, int i2) {
        HashMap hashMap = new HashMap();
        hashMap.put(NativeProtocol.WEB_DIALOG_ACTION, "setSpeed");
        hashMap.put("funId", 2);
        hashMap.put("speed", Integer.valueOf(i));
        hashMap.put("onlyRead", Integer.valueOf(i2));
        send(this.mGson.toJson(hashMap));
    }

    public void setVibrate(int i, JsonObject jsonObject) {
        HashMap hashMap = new HashMap();
        hashMap.put(NativeProtocol.WEB_DIALOG_ACTION, "setSpeed");
        hashMap.put("funId", 2);
        hashMap.put("speed", Integer.valueOf(i));
        hashMap.put("lines", jsonObject);
        send(this.mGson.toJson(hashMap));
    }

    public void setVibrateMode(int i) {
        setVibrateMode(i, 0);
    }

    public void setVibrateMode(int i, int i2) {
        HashMap hashMap = new HashMap();
        hashMap.put(NativeProtocol.WEB_DIALOG_ACTION, "setMode");
        hashMap.put("funId", 2);
        hashMap.put("mode", Integer.valueOf(i));
        hashMap.put("onlyRead", Integer.valueOf(i2));
        send(this.mGson.toJson(hashMap));
    }

    public void switchMode(int i) {
        HashMap hashMap = new HashMap();
        hashMap.put(NativeProtocol.WEB_DIALOG_ACTION, "switchMode");
        hashMap.put("status", Integer.valueOf(i));
        send(this.mGson.toJson(hashMap));
    }

    public void worldMessage(String str) {
        HashMap hashMap = new HashMap();
        hashMap.put(NativeProtocol.WEB_DIALOG_ACTION, "worldMessage");
        hashMap.put(ShareConstants.WEB_DIALOG_PARAM_MESSAGE, str);
        send(this.mGson.toJson(hashMap));
    }

    public void worldMessageToUser(int i, String str) {
        HashMap hashMap = new HashMap();
        hashMap.put(NativeProtocol.WEB_DIALOG_ACTION, "worldMessage");
        hashMap.put("userTo", Integer.valueOf(i));
        hashMap.put(ShareConstants.WEB_DIALOG_PARAM_MESSAGE, str);
        send(this.mGson.toJson(hashMap));
    }

    public void worldRooms(boolean z) {
        String deviceCode = MyApplication.getApplication().getDeviceCode();
        if (TextUtils.isEmpty(deviceCode) && this.onSocketNotifyListener != null) {
            this.onSocketNotifyListener.onCodeNull();
        }
        HashMap hashMap = new HashMap();
        hashMap.put(NativeProtocol.WEB_DIALOG_ACTION, "worldRooms");
        hashMap.put("productId", deviceCode);
        hashMap.put("new", Boolean.valueOf(z));
        send(this.mGson.toJson(hashMap));
    }

    public void worldUsers(boolean z) {
        HashMap hashMap = new HashMap();
        hashMap.put(NativeProtocol.WEB_DIALOG_ACTION, "worldUsers");
        hashMap.put("new", Boolean.valueOf(z));
        send(this.mGson.toJson(hashMap));
    }
}
