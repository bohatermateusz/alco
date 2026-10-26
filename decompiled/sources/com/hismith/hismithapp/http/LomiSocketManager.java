package com.hismith.hismithapp.http;

import android.content.Context;
import android.content.Intent;
import android.graphics.Point;
import android.os.Handler;
import android.os.Message;
import android.text.TextUtils;
import com.facebook.internal.NativeProtocol;
import com.facebook.share.internal.ShareConstants;
import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.hismith.hismithapp.LoginActivity;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.bluetooth.BleManager;
import com.hismith.hismithapp.model.FunDatModel;
import com.hismith.hismithapp.model.ProductModel;
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
import java.util.ArrayList;
import java.util.HashMap;
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
public class LomiSocketManager {
    public static final int SOCKET_CONNECTED = 1;
    public static final int SOCKET_DISCONNECTED = 2;
    public static final int SOCKET_ERROR = 3;
    public static final int SOCKET_HEARTBEAT = 6;
    public static final int SOCKET_RECONNECT = 5;
    public static final int SOCKET_RESPONSE = 4;
    public static final int TYPE_CONN_AI = 6;
    private static LomiSocketManager instance = null;
    private static final int mHeartBeatDelay = 40000;
    private static WebSocket mWebSocket;
    private OkHttpClient httpClient;
    private Context mContext;
    private volatile OnLomiSocketNotifyListener onSocketNotifyListener;
    private int mWidth = 0;
    private int mHeight = 0;
    private int connType = 0;
    private boolean isFailure = false;
    private boolean isConnect = false;
    private int timeout = 90000;
    private Handler mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.http.LomiSocketManager.1
        @Override // android.os.Handler.Callback
        public boolean handleMessage(Message message) {
            int i = message.what;
            if (i == 3) {
                ToastUtil.showShortToastCenter(LomiSocketManager.this.mContext, LanguageUtil.getLanguageByKey(LomiSocketManager.this.mContext, "networkConnectionTimedOut"));
                return false;
            }
            if (i == 4) {
                LomiSocketManager.this.receResponse((String) message.obj);
                return false;
            }
            if (i != 6) {
                return false;
            }
            LomiSocketManager.this.heartbeat();
            LomiSocketManager.this.mHandler.sendEmptyMessageDelayed(6, 40000L);
            return false;
        }
    });
    private Gson mGson = new Gson();
    private RetrofitRequest baseRequest = new RetrofitRequest();

    public interface OnLomiSocketNotifyListener {
        void OnNotify(String str);

        void onClosed(int i, String str);

        void onCodeNull();

        void onFailure(Throwable th, Response response);

        void onOpen();
    }

    public LomiSocketManager(Context context) {
        this.mContext = context;
    }

    private WebSocketListener createListener() {
        return new WebSocketListener() { // from class: com.hismith.hismithapp.http.LomiSocketManager.2
            @Override // okhttp3.WebSocketListener
            public void onClosed(WebSocket webSocket, int i, String str) {
                super.onClosed(webSocket, i, str);
                LogUtil.e("WebSocket: onClosed code:" + i + ", reason:" + str);
                if (webSocket == LomiSocketManager.mWebSocket || LomiSocketManager.mWebSocket == null) {
                    WebSocket unused = LomiSocketManager.mWebSocket = null;
                    LomiSocketManager.this.isConnect = false;
                    LomiSocketManager.this.isFailure = true;
                    LomiSocketManager.this.clearHandler();
                    if (LomiSocketManager.this.onSocketNotifyListener != null) {
                        LomiSocketManager.this.onSocketNotifyListener.onClosed(i, str);
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
                LogUtil.e("WebSocket: onFailure：" + stringWriter.toString() + ", webSocket:" + webSocket + ", mWebSocket:" + LomiSocketManager.mWebSocket + ", 是否和当前webSocket相等" + (webSocket == LomiSocketManager.mWebSocket));
                if (webSocket == LomiSocketManager.mWebSocket || LomiSocketManager.mWebSocket == null) {
                    WebSocket unused = LomiSocketManager.mWebSocket = null;
                    LomiSocketManager.this.isConnect = false;
                    LomiSocketManager.this.isFailure = true;
                    LomiSocketManager.this.clearHandler();
                    if ((th instanceof ConnectException) || (th instanceof SocketTimeoutException) || (th instanceof UnknownHostException)) {
                        LomiSocketManager.this.mHandler.sendEmptyMessage(3);
                    }
                    LogUtil.e("WebSocket: onFailure：" + th.getClass());
                    if (LomiSocketManager.this.onSocketNotifyListener != null) {
                        LomiSocketManager.this.onSocketNotifyListener.onFailure(th, response);
                    }
                }
            }

            @Override // okhttp3.WebSocketListener
            public void onMessage(WebSocket webSocket, String str) {
                super.onMessage(webSocket, str);
                if (LomiSocketManager.mWebSocket == null) {
                    WebSocket unused = LomiSocketManager.mWebSocket = webSocket;
                }
                LomiSocketManager.this.isConnect = true;
                LomiSocketManager.this.isFailure = false;
                Message message = new Message();
                message.what = 4;
                message.obj = str;
                LomiSocketManager.this.mHandler.sendMessage(message);
            }

            @Override // okhttp3.WebSocketListener
            public void onMessage(WebSocket webSocket, ByteString byteString) {
                super.onMessage(webSocket, byteString);
                LomiSocketManager.this.isConnect = true;
            }

            @Override // okhttp3.WebSocketListener
            public void onOpen(WebSocket webSocket, Response response) {
                super.onOpen(webSocket, response);
                LomiSocketManager.this.isConnect = true;
                LomiSocketManager.this.close();
                LomiSocketManager.this.closeSocket();
                LogUtil.e("WebSocket: onOpen:" + webSocket + ", type:" + LomiSocketManager.this.connType);
                WebSocket unused = LomiSocketManager.mWebSocket = webSocket;
                if (LomiSocketManager.this.connType == 6) {
                    LomiSocketManager.this.helloLomi();
                }
                if (LomiSocketManager.this.onSocketNotifyListener != null) {
                    LomiSocketManager.this.onSocketNotifyListener.onOpen();
                }
            }
        };
    }

    public static LomiSocketManager getInstance() {
        LomiSocketManager lomiSocketManager;
        synchronized (LomiSocketManager.class) {
            lomiSocketManager = instance;
            if (lomiSocketManager == null) {
                lomiSocketManager = new LomiSocketManager(MyApplication.getApplication());
                instance = lomiSocketManager;
            }
        }
        return lomiSocketManager;
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
    public void receResponse(String str) {
        if (TextUtils.isEmpty(str)) {
            return;
        }
        LogUtil.e("WebSocket " + mWebSocket + " : 接收:" + modifyDatField("dat", str));
        try {
            JsonObject asJsonObject = new JsonParser().parse(str).getAsJsonObject();
            if ((asJsonObject.get("success") == null ? 0 : asJsonObject.get("success").getAsInt()) == 0) {
                String str2 = "";
                String asString = asJsonObject.get("errorCode") == null ? "" : asJsonObject.get("errorCode").getAsString();
                if (asJsonObject.get("errorMessage") != null) {
                    str2 = asJsonObject.get("errorMessage").getAsString();
                }
                if (!TextUtils.isEmpty(asString)) {
                    if (!TextUtils.isEmpty(str2)) {
                        ToastUtil.showShortToastCenter(this.mContext, str2);
                    }
                    if (Constans.errorRelogin.equals(asString)) {
                        String str3 = (String) SharedPreferencesUtil.get(this.mContext, Constans.Language);
                        Object obj = SharedPreferencesUtil.get(this.mContext, Constans.contactUserId, 0);
                        int intValue = obj != null ? ((Integer) obj).intValue() : 0;
                        String str4 = (String) SharedPreferencesUtil.get(this.mContext, Constans.storeUrl);
                        String str5 = (String) SharedPreferencesUtil.get(this.mContext, Constans.roomVoice);
                        String str6 = (String) SharedPreferencesUtil.get(this.mContext, Constans.tempToken);
                        String str7 = (String) SharedPreferencesUtil.get(this.mContext, Constans.aboutApp);
                        int i = intValue;
                        String str8 = (String) SharedPreferencesUtil.get(this.mContext, Constans.copyright);
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
            }
            if (this.onSocketNotifyListener != null) {
                this.onSocketNotifyListener.OnNotify(str);
            }
        } catch (Exception e2) {
            e2.printStackTrace();
        }
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
        WebSocket webSocket = mWebSocket;
        if (webSocket != null) {
            webSocket.close(1000, "");
            mWebSocket = null;
        }
    }

    public void connSocket(int i) {
        this.connType = i;
        this.httpClient = new OkHttpClient.Builder().connectTimeout(0L, TimeUnit.MILLISECONDS).dns(new XDns(this.timeout)).build();
        String str = this.connType == 6 ? "ws://lomi.hismiths.com:8888" : "";
        LogUtil.e("connSocket:".concat(str));
        this.httpClient.newWebSocket(new Request.Builder().url(str).build(), createListener());
        this.isFailure = false;
        this.isConnect = false;
        LogUtil.e("WebSocket 连接Socket");
    }

    public OnLomiSocketNotifyListener getOnSocketNotifyListener() {
        return this.onSocketNotifyListener;
    }

    public WebSocket getWebSocket() {
        return mWebSocket;
    }

    public void heartbeat() {
        HashMap hashMap = new HashMap();
        hashMap.put(NativeProtocol.WEB_DIALOG_ACTION, "heartbeat");
        send(this.mGson.toJson(hashMap));
    }

    public void helloLomi() {
        String str = (String) SharedPreferencesUtil.get(this.mContext, Constans.Token);
        String deviceCode = MyApplication.getApplication().getDeviceCode();
        ProductModel productModel = MyApplication.getApplication().getmProductModel();
        if (TextUtils.isEmpty(deviceCode) || productModel == null) {
            if (this.onSocketNotifyListener != null) {
                this.onSocketNotifyListener.onCodeNull();
                return;
            }
            return;
        }
        ArrayList arrayList = new ArrayList();
        for (int i = 0; i < productModel.getFuns().length; i++) {
            FunDatModel funDatModel = new FunDatModel();
            funDatModel.setFunId(productModel.getFuns()[i].getfId());
            if (BleManager.getInstance(this.mContext).getmConnBleDevice() != null) {
                if (i == 0) {
                    funDatModel.setSpeed(BleManager.getInstance(this.mContext).getmConnBleDevice().getSpeed());
                    funDatModel.setMode(BleManager.getInstance(this.mContext).getmConnBleDevice().getMode());
                } else if (i == 1) {
                    funDatModel.setSpeed(BleManager.getInstance(this.mContext).getmConnBleDevice().getVibration());
                    funDatModel.setMode(BleManager.getInstance(this.mContext).getmConnBleDevice().getVibrationMode());
                }
            }
            arrayList.add(funDatModel);
        }
        HashMap hashMap = new HashMap();
        hashMap.put(Constans.Token, str);
        hashMap.put(NativeProtocol.WEB_DIALOG_ACTION, "helloLomi");
        hashMap.put("productId", deviceCode);
        hashMap.put("appBrand", "");
        hashMap.put("funDat", arrayList.toArray());
        send(this.mGson.toJson(hashMap));
    }

    public boolean isConnect() {
        return this.isConnect;
    }

    public boolean lomiTxt(String str, long j) {
        HashMap hashMap = new HashMap();
        hashMap.put(NativeProtocol.WEB_DIALOG_ACTION, "lomiTxt");
        hashMap.put(ShareConstants.WEB_DIALOG_PARAM_MESSAGE, str);
        hashMap.put("timestamp", Long.valueOf(j));
        return send(this.mGson.toJson(hashMap));
    }

    public boolean lomiVoice(String str) {
        HashMap hashMap = new HashMap();
        hashMap.put(NativeProtocol.WEB_DIALOG_ACTION, "lomiVoice");
        hashMap.put("voiceData", str);
        return send(this.mGson.toJson(hashMap));
    }

    public boolean send(String str) {
        if (this.isFailure) {
            LogUtil.e("WebSocket isFailure");
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

    public void setClimax() {
        HashMap hashMap = new HashMap();
        hashMap.put(NativeProtocol.WEB_DIALOG_ACTION, "setClimax");
        send(this.mGson.toJson(hashMap));
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

    public void setOnSocketNotifyListener(OnLomiSocketNotifyListener onLomiSocketNotifyListener) {
        this.onSocketNotifyListener = onLomiSocketNotifyListener;
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

    public LomiSocketManager setScreenSize(int i, int i2) {
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
}
