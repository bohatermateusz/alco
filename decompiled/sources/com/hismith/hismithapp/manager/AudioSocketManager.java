package com.hismith.hismithapp.manager;

import android.content.Context;
import android.os.Handler;
import android.os.Message;
import android.text.TextUtils;
import com.facebook.appevents.internal.ViewHierarchyConstants;
import com.facebook.internal.NativeProtocol;
import com.facebook.share.internal.ShareConstants;
import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.http.RetrofitRequest;
import com.hismith.hismithapp.http.SocketSendModel;
import com.hismith.hismithapp.http.XDns;
import com.hismith.hismithapp.model.LineRoomInfoModel;
import com.hismith.hismithapp.model.LineRoomMessageModel;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import java.util.HashMap;
import java.util.concurrent.TimeUnit;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;
import okhttp3.WebSocket;
import okhttp3.WebSocketListener;
import okio.ByteString;

/* loaded from: classes3.dex */
public class AudioSocketManager {
    public static final int SOCKET_CONNECTED = 1;
    public static final int SOCKET_DISCONNECTED = 2;
    public static final int SOCKET_ERROR = 3;
    public static final int SOCKET_HEARTBEAT = 6;
    public static final int SOCKET_RECONNECT = 5;
    public static final int SOCKET_RESPONSE = 4;
    public static final int TYPE_CONN_AUDIO = 3;
    public static final int TYPE_CONN_LINE = 2;
    public static final int TYPE_CONN_REMOTE = 1;
    private static AudioSocketManager instance = null;
    private static final int mHeartBeatDelay = 60000;
    public static final String welcomeMsgFlag = "*#*";
    private OkHttpClient httpClient;
    private final Context mContext;
    private WebSocket mWebSocket;
    private OnAudioSocketNotifyListener onAudioSocketNotifyListener;
    private LineRoomInfoModel roomInfoModel;
    private LineRoomMessageModel roomMessageModel;
    private int mWidth = 0;
    private int mHeight = 0;
    private int connType = 0;
    private boolean isColseSound = true;
    private final int roomId = 0;
    private final int roomIsPublic = 1;
    private boolean isFailure = false;
    private final int timeout = 60000;
    private final Handler mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.manager.AudioSocketManager.1
        @Override // android.os.Handler.Callback
        public boolean handleMessage(Message message) {
            int i = message.what;
            if (i == 4) {
                AudioSocketManager.this.receResponse((String) message.obj);
                return false;
            }
            if (i != 6) {
                return false;
            }
            AudioSocketManager.this.heartbeat();
            AudioSocketManager.this.mHandler.sendEmptyMessageDelayed(6, 60000L);
            return false;
        }
    });
    private final Gson mGson = new Gson();
    private final RetrofitRequest baseRequest = new RetrofitRequest();

    public interface OnAudioSocketNotifyListener {
        void OnAudioNotify(String str);

        void onAudioClosed(int i, String str);

        void onAudioFailure(Throwable th, Response response);

        void onAudioOpen();
    }

    public AudioSocketManager(Context context) {
        this.mContext = context;
    }

    private WebSocketListener createListener() {
        return new WebSocketListener() { // from class: com.hismith.hismithapp.manager.AudioSocketManager.2
            @Override // okhttp3.WebSocketListener
            public void onClosed(WebSocket webSocket, int i, String str) {
                super.onClosed(webSocket, i, str);
                LogUtil.e("AudioWebSocketWebSocket: onClosed:" + webSocket + ", mWebSocket:" + AudioSocketManager.this.mWebSocket);
                if (webSocket == AudioSocketManager.this.mWebSocket || AudioSocketManager.this.mWebSocket == null) {
                    AudioSocketManager.this.mWebSocket = null;
                    AudioSocketManager.this.isFailure = true;
                    AudioSocketManager.this.clearHandler();
                    if (AudioSocketManager.this.onAudioSocketNotifyListener != null) {
                        AudioSocketManager.this.onAudioSocketNotifyListener.onAudioClosed(i, str);
                    }
                }
            }

            @Override // okhttp3.WebSocketListener
            public void onClosing(WebSocket webSocket, int i, String str) {
                super.onClosing(webSocket, i, str);
            }

            @Override // okhttp3.WebSocketListener
            public void onFailure(WebSocket webSocket, Throwable th, Response response) {
                super.onFailure(webSocket, th, response);
                LogUtil.e("AudioWebSocketWebSocket: onFailure：" + th.getMessage() + ", webSocket:" + webSocket + ", mWebSocket:" + AudioSocketManager.this.mWebSocket);
                if (webSocket == AudioSocketManager.this.mWebSocket || AudioSocketManager.this.mWebSocket == null) {
                    AudioSocketManager.this.mWebSocket = null;
                    AudioSocketManager.this.isFailure = true;
                    AudioSocketManager.this.clearHandler();
                    if (AudioSocketManager.this.onAudioSocketNotifyListener != null) {
                        AudioSocketManager.this.onAudioSocketNotifyListener.onAudioFailure(th, response);
                    }
                }
            }

            @Override // okhttp3.WebSocketListener
            public void onMessage(WebSocket webSocket, String str) {
                super.onMessage(webSocket, str);
                AudioSocketManager.this.receResponse(str);
            }

            @Override // okhttp3.WebSocketListener
            public void onMessage(WebSocket webSocket, ByteString byteString) {
                super.onMessage(webSocket, byteString);
                LogUtil.e("AudioWebSocketWebSocket onMessagebyte" + AudioSocketManager.this.mWebSocket + " : 接收:" + byteString);
            }

            @Override // okhttp3.WebSocketListener
            public void onOpen(WebSocket webSocket, Response response) {
                super.onOpen(webSocket, response);
                AudioSocketManager.this.closeSocket();
                LogUtil.e("AudioWebSocketWebSocket: onOpen:" + webSocket);
                AudioSocketManager.this.mWebSocket = webSocket;
                int i = AudioSocketManager.this.connType;
                AudioSocketManager audioSocketManager = AudioSocketManager.this;
                if (i == 1) {
                    audioSocketManager.getRemoteUrl();
                } else {
                    int i2 = audioSocketManager.connType;
                    AudioSocketManager audioSocketManager2 = AudioSocketManager.this;
                    if (i2 == 2) {
                        audioSocketManager2.onLinePlay();
                    } else if (audioSocketManager2.connType == 3) {
                        boolean z = AudioSocketManager.this.isColseSound;
                        AudioSocketManager audioSocketManager3 = AudioSocketManager.this;
                        if (z) {
                            audioSocketManager3.closeSound();
                        } else {
                            audioSocketManager3.openSound();
                        }
                    }
                }
                if (AudioSocketManager.this.onAudioSocketNotifyListener != null) {
                    AudioSocketManager.this.onAudioSocketNotifyListener.onAudioOpen();
                }
            }
        };
    }

    public static AudioSocketManager getInstance() {
        AudioSocketManager audioSocketManager;
        synchronized (AudioSocketManager.class) {
            audioSocketManager = instance;
            if (audioSocketManager == null) {
                audioSocketManager = new AudioSocketManager(MyApplication.getApplication());
                instance = audioSocketManager;
            }
        }
        return audioSocketManager;
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void receResponse(String str) {
        if (TextUtils.isEmpty(str)) {
            return;
        }
        LogUtil.e("AudioWebSocketWebSocket " + this.mWebSocket + " : 接收:" + str);
        try {
            OnAudioSocketNotifyListener onAudioSocketNotifyListener = this.onAudioSocketNotifyListener;
            if (onAudioSocketNotifyListener != null) {
                onAudioSocketNotifyListener.OnAudioNotify(str);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void agreeUserRoom(String str) {
        HashMap hashMap = new HashMap();
        hashMap.put(NativeProtocol.WEB_DIALOG_ACTION, "agreeUserRoom");
        hashMap.put("userId", str);
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
        WebSocket webSocket = this.mWebSocket;
        if (webSocket != null) {
            webSocket.close(1000, "");
        }
    }

    public void closeSound() {
        String str = (String) SharedPreferencesUtil.get(this.mContext, Constans.Token);
        HashMap hashMap = new HashMap();
        hashMap.put(Constans.Token, str);
        hashMap.put("appBrand", "");
        hashMap.put(NativeProtocol.WEB_DIALOG_ACTION, "closeSound");
        send(this.mGson.toJson(hashMap));
    }

    public void closeVoice() {
        String str = (String) SharedPreferencesUtil.get(this.mContext, Constans.Token);
        HashMap hashMap = new HashMap();
        hashMap.put(Constans.Token, str);
        hashMap.put("appBrand", "");
        hashMap.put(NativeProtocol.WEB_DIALOG_ACTION, "closeVoice");
        send(this.mGson.toJson(hashMap));
    }

    public void connSocket(int i, boolean z) {
        this.connType = i;
        this.isColseSound = z;
        this.httpClient = new OkHttpClient.Builder().connectTimeout(0L, TimeUnit.MILLISECONDS).dns(new XDns(60000L)).build();
        LogUtil.e("connSocket:ws://www.hismiths.com:8889");
        this.httpClient.newWebSocket(new Request.Builder().url("ws://www.hismiths.com:8889").build(), createListener());
        this.isFailure = false;
        LogUtil.e("AudioWebSocket 连接Socket");
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

    public void getRemoteUrl() {
        send(this.mGson.toJson(new SocketSendModel((String) SharedPreferencesUtil.get(this.mContext, Constans.Token), "", "getRemoteUrl", this.mWidth, this.mHeight, MyApplication.getApplication().getDeviceCode(), MyApplication.getApplication().getDeviceName())));
    }

    public int getRoomId() {
        return 0;
    }

    public LineRoomInfoModel getRoomInfoModel() {
        return this.roomInfoModel;
    }

    public int getRoomIsPublic() {
        return 1;
    }

    public LineRoomMessageModel getRoomMessageModel() {
        return this.roomMessageModel;
    }

    public void getRooms(int i) {
        String deviceCode = MyApplication.getApplication().getDeviceCode();
        HashMap hashMap = new HashMap();
        hashMap.put(NativeProtocol.WEB_DIALOG_ACTION, "getRooms");
        hashMap.put("productId", deviceCode);
        hashMap.put("new", Integer.valueOf(i));
        send(this.mGson.toJson(hashMap));
    }

    public WebSocket getWebSocket() {
        return this.mWebSocket;
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

    public void inviteUserRoom(int i) {
        HashMap hashMap = new HashMap();
        hashMap.put(NativeProtocol.WEB_DIALOG_ACTION, "inviteUserRoom");
        hashMap.put("userId", Integer.valueOf(i));
        send(this.mGson.toJson(hashMap));
    }

    public void onLinePlay() {
        String str = (String) SharedPreferencesUtil.get(this.mContext, Constans.Token);
        String deviceCode = MyApplication.getApplication().getDeviceCode();
        HashMap hashMap = new HashMap();
        hashMap.put(Constans.Token, str);
        hashMap.put("appBrand", "");
        hashMap.put(NativeProtocol.WEB_DIALOG_ACTION, "onlinePlay");
        hashMap.put("productId", deviceCode);
        hashMap.put(ViewHierarchyConstants.DIMENSION_WIDTH_KEY, Integer.valueOf(this.mWidth));
        hashMap.put(ViewHierarchyConstants.DIMENSION_HEIGHT_KEY, Integer.valueOf(this.mHeight));
        send(this.mGson.toJson(hashMap));
    }

    public void openSound() {
        String str = (String) SharedPreferencesUtil.get(this.mContext, Constans.Token);
        HashMap hashMap = new HashMap();
        hashMap.put(Constans.Token, str);
        hashMap.put("appBrand", "");
        hashMap.put(NativeProtocol.WEB_DIALOG_ACTION, "openSound");
        send(this.mGson.toJson(hashMap));
    }

    public boolean openVoice() {
        String str = (String) SharedPreferencesUtil.get(this.mContext, Constans.Token);
        HashMap hashMap = new HashMap();
        hashMap.put(Constans.Token, str);
        hashMap.put("appBrand", "");
        hashMap.put(NativeProtocol.WEB_DIALOG_ACTION, "openVoice");
        return send(this.mGson.toJson(hashMap));
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

    public void roomUser() {
        HashMap hashMap = new HashMap();
        hashMap.put(NativeProtocol.WEB_DIALOG_ACTION, "roomUser");
        send(this.mGson.toJson(hashMap));
    }

    public void roomVoice(String str) {
        HashMap hashMap = new HashMap();
        hashMap.put(NativeProtocol.WEB_DIALOG_ACTION, Constans.roomVoice);
        hashMap.put("voiceData", str);
        send(this.mGson.toJson(hashMap));
    }

    public boolean send(String str) {
        if (this.isFailure) {
            LogUtil.e("AudioWebSocketWebSocket isFailure");
            OnAudioSocketNotifyListener onAudioSocketNotifyListener = this.onAudioSocketNotifyListener;
            if (onAudioSocketNotifyListener != null) {
                onAudioSocketNotifyListener.onAudioClosed(0, "NULL");
            }
        } else {
            WebSocket webSocket = this.mWebSocket;
            if (webSocket != null) {
                LogUtil.e("AudioWebSocketWebSocket " + this.mWebSocket + " 发送：" + str + ", flag:" + webSocket.send(str));
                if (this.mHandler.hasMessages(6)) {
                    this.mHandler.removeMessages(6);
                }
                this.mHandler.sendEmptyMessageDelayed(6, 60000L);
                return true;
            }
        }
        return false;
    }

    public void setOnAudioSocketNotifyListener(OnAudioSocketNotifyListener onAudioSocketNotifyListener) {
        this.onAudioSocketNotifyListener = onAudioSocketNotifyListener;
    }

    public void setRoomMessageModel(LineRoomMessageModel lineRoomMessageModel) {
        this.roomMessageModel = lineRoomMessageModel;
    }

    public AudioSocketManager setScreenSize(int i, int i2) {
        this.mWidth = i;
        this.mHeight = i2;
        return instance;
    }

    public void setSpeed(int i, JsonObject jsonObject) {
        HashMap hashMap = new HashMap();
        hashMap.put(NativeProtocol.WEB_DIALOG_ACTION, "setSpeed");
        hashMap.put("speed", Integer.valueOf(i));
        hashMap.put("lines", jsonObject);
        send(this.mGson.toJson(hashMap));
    }
}
