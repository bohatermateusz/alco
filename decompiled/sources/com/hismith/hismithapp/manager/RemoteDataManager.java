package com.hismith.hismithapp.manager;

import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.graphics.Point;
import android.os.Handler;
import android.os.Message;
import android.text.TextUtils;
import android.view.View;
import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.bluetooth.BleManager;
import com.hismith.hismithapp.floating.FloatBase;
import com.hismith.hismithapp.floating.FloatManager;
import com.hismith.hismithapp.http.SocketManager;
import com.hismith.hismithapp.model.RemotePlayModel;
import com.hismith.hismithapp.observer.AppBackgroundUtil;
import com.hismith.hismithapp.product.ProductUtil;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.ModeDataVCUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import java.util.HashMap;
import java.util.Map;
import okhttp3.Response;

/* loaded from: classes3.dex */
public class RemoteDataManager implements SocketManager.OnSocketNotifyListener, BleManager.OnNotifyChangeListener_v2 {
    private static final int WHAT_AUTO = 18;
    private static final int WHAT_CLIENT_CONNECTION = 5;
    private static final int WHAT_CLIENT_OFFLINE = 6;
    private static final int WHAT_CLIMAX = 7;
    private static final int WHAT_CLOSE = 9;
    private static final int WHAT_CODE_NULL = 10;
    private static final int WHAT_END_POSITION = 12;
    private static final int WHAT_LINE = 19;
    private static final int WHAT_LINE_VIBRATE = 20;
    private static final int WHAT_MESSAGE = 25;
    private static final int WHAT_MODE = 2;
    private static final int WHAT_MUSIC = 17;
    private static final int WHAT_NOTIFY = 15;
    private static final int WHAT_OPEN = 8;
    private static final int WHAT_PIU = 24;
    private static final int WHAT_POSITION = 14;
    private static final int WHAT_REVERSAL_ROTATION = 23;
    private static final int WHAT_SHUTDOWN = 16;
    private static final int WHAT_SMOOTH = 13;
    private static final int WHAT_SPEED = 1;
    private static final int WHAT_START_POSITION = 11;
    private static final int WHAT_VIBRATE = 3;
    private static final int WHAT_VIBRATE_MODE = 4;
    private static final int WHAT_WEB_SPEED = 21;
    private static final int WHAT_WEB_VIBRATE = 22;
    private static RemoteDataManager instance;
    private Callback callback;
    private int limitMax;
    private int limitVibrateMax;
    private final Context mContext;
    private FloatBase mFloatBase;
    private final Gson gson = new Gson();
    private int limitStartPosition = 0;
    private int limitStopPosition = 100;
    private int mSpeed = 0;
    private int mVibrate = 0;
    private final Handler mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.manager.RemoteDataManager.1
        /* JADX WARN: Can't fix incorrect switch cases order, some code will duplicate */
        /* JADX WARN: Code restructure failed: missing block: B:108:0x030a, code lost:
        
            return false;
         */
        @Override // android.os.Handler.Callback
        /*
            Code decompiled incorrectly, please refer to instructions dump.
        */
        public boolean handleMessage(Message message) {
            Map<String, Point> map = null;
            switch (message.what) {
                case 1:
                    RemoteDataManager.this.mSpeed = message.arg1;
                    if (RemoteDataManager.this.mFloatBase != null) {
                        RemoteDataManager.this.mFloatBase.refreshFloatSpeedUI(RemoteDataManager.this.mSpeed);
                    }
                    if (RemoteDataManager.this.callback != null) {
                        RemoteDataManager.this.callback.onSpeedUI(RemoteDataManager.this.mSpeed);
                        break;
                    }
                    break;
                case 2:
                    int i = message.arg1;
                    if (RemoteDataManager.this.mFloatBase != null) {
                        RemoteDataManager.this.mFloatBase.refreshFloatModeUI(i);
                    }
                    if (RemoteDataManager.this.callback != null) {
                        RemoteDataManager.this.callback.onSpeedModeUI(i);
                        break;
                    }
                    break;
                case 3:
                    RemoteDataManager.this.mVibrate = message.arg1;
                    if (RemoteDataManager.this.mFloatBase != null) {
                        RemoteDataManager.this.mFloatBase.refreshFloatVibrateUI(RemoteDataManager.this.mVibrate);
                    }
                    if (RemoteDataManager.this.callback != null) {
                        RemoteDataManager.this.callback.onVibrateUI(RemoteDataManager.this.mVibrate);
                        break;
                    }
                    break;
                case 4:
                    int i2 = message.arg1;
                    if (RemoteDataManager.this.mFloatBase != null) {
                        RemoteDataManager.this.mFloatBase.refreshFloatVibrateModeUI(i2);
                    }
                    if (RemoteDataManager.this.callback != null) {
                        RemoteDataManager.this.callback.onVibrateModeUI(i2);
                        break;
                    }
                    break;
                case 5:
                    if (RemoteDataManager.this.callback != null) {
                        RemoteDataManager.this.callback.onClientConnection();
                        break;
                    }
                    break;
                case 6:
                    if (RemoteDataManager.this.callback != null) {
                        RemoteDataManager.this.callback.onClientOffline();
                        break;
                    }
                    break;
                case 7:
                    if (RemoteDataManager.this.mFloatBase != null) {
                        RemoteDataManager.this.mFloatBase.refreshFloatClimaxUI(0);
                    }
                    if (RemoteDataManager.this.callback != null) {
                        RemoteDataManager.this.callback.onOpenClimax();
                        break;
                    }
                    break;
                case 8:
                    if (RemoteDataManager.this.callback != null) {
                        RemoteDataManager.this.callback.onOpen();
                        break;
                    }
                    break;
                case 9:
                    if (RemoteDataManager.this.callback != null) {
                        RemoteDataManager.this.callback.onClose();
                        break;
                    }
                    break;
                case 10:
                    if (RemoteDataManager.this.callback != null) {
                        RemoteDataManager.this.callback.onCodeNull();
                        break;
                    }
                    break;
                case 11:
                    int i3 = message.arg1;
                    if (RemoteDataManager.this.mFloatBase != null) {
                        RemoteDataManager.this.mFloatBase.refreshFloatStartPosition(i3);
                    }
                    if (RemoteDataManager.this.callback != null) {
                        RemoteDataManager.this.callback.onStartPositionUI(i3);
                        break;
                    }
                    break;
                case 12:
                    int i4 = message.arg1;
                    if (RemoteDataManager.this.mFloatBase != null) {
                        RemoteDataManager.this.mFloatBase.refreshFloatStopPosition(i4);
                    }
                    if (RemoteDataManager.this.callback != null) {
                        RemoteDataManager.this.callback.onEndPositionUI(i4);
                        break;
                    }
                    break;
                case 13:
                    int i5 = message.arg1;
                    if (RemoteDataManager.this.mFloatBase != null) {
                        RemoteDataManager.this.mFloatBase.refreshFloatSmooth(i5);
                    }
                    if (RemoteDataManager.this.callback != null) {
                        RemoteDataManager.this.callback.onSmoothUI(i5);
                        break;
                    }
                    break;
                case 14:
                    int i6 = message.arg1;
                    if (RemoteDataManager.this.mFloatBase != null) {
                        RemoteDataManager.this.mFloatBase.refreshFloatPosition(i6);
                    }
                    if (RemoteDataManager.this.callback != null) {
                        RemoteDataManager.this.callback.onPositionUI(i6);
                        break;
                    }
                    break;
                case 15:
                    if (RemoteDataManager.this.callback != null) {
                        RemoteDataManager.this.callback.onNotify();
                        break;
                    }
                    break;
                case 16:
                    if (RemoteDataManager.this.callback != null) {
                        RemoteDataManager.this.callback.onSpeedUI(0);
                    }
                    if (RemoteDataManager.this.callback != null) {
                        RemoteDataManager.this.callback.onVibrateUI(0);
                    }
                    if (RemoteDataManager.this.mFloatBase != null) {
                        RemoteDataManager.this.mFloatBase.refreshFloatShutdown();
                        break;
                    }
                    break;
                case 17:
                    int i7 = message.arg1;
                    if (RemoteDataManager.this.mFloatBase != null) {
                        RemoteDataManager.this.mFloatBase.refreshFloatMusicUI(i7);
                        break;
                    }
                    break;
                case 18:
                    int i8 = message.arg1;
                    if (RemoteDataManager.this.mFloatBase != null) {
                        RemoteDataManager.this.mFloatBase.refreshFloatRunMode(i8);
                    }
                    if (RemoteDataManager.this.callback != null) {
                        RemoteDataManager.this.callback.onAutoUI(i8);
                        break;
                    }
                    break;
                case 19:
                    try {
                        map = (Map) message.obj;
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                    if (RemoteDataManager.this.callback != null) {
                        RemoteDataManager.this.callback.onLinePoints(map);
                        break;
                    }
                    break;
                case 20:
                    try {
                        map = (Map) message.obj;
                    } catch (Exception e2) {
                        e2.printStackTrace();
                    }
                    if (RemoteDataManager.this.callback != null) {
                        RemoteDataManager.this.callback.onVibrateLinePoints(map);
                        break;
                    }
                    break;
                case 21:
                    int i9 = message.arg1;
                    if (RemoteDataManager.this.callback != null) {
                        RemoteDataManager.this.callback.onWebSpeedUI(i9);
                        break;
                    }
                    break;
                case 22:
                    int i10 = message.arg1;
                    if (RemoteDataManager.this.callback != null) {
                        RemoteDataManager.this.callback.onWebVibrateUI(i10);
                        break;
                    }
                    break;
                case 23:
                    int i11 = message.arg1;
                    if (RemoteDataManager.this.callback != null) {
                        RemoteDataManager.this.callback.onReversalRotation(i11);
                        break;
                    }
                    break;
                case 24:
                    int intValue = ((Integer) message.obj).intValue();
                    if (RemoteDataManager.this.callback != null) {
                        RemoteDataManager.this.callback.onPiuUI(intValue);
                        break;
                    }
                    break;
                case 25:
                    String obj = message.obj.toString();
                    if (RemoteDataManager.this.callback != null) {
                        RemoteDataManager.this.callback.onMessage(obj);
                        break;
                    }
                    break;
            }
        }
    });

    public interface Callback {
        void onAutoUI(int i);

        void onClientConnection();

        void onClientOffline();

        void onClose();

        void onCodeNull();

        void onEndPositionUI(int i);

        void onLinePoints(Map<String, Point> map);

        void onMessage(String str);

        void onNotify();

        void onOpen();

        void onOpenClimax();

        void onPiuUI(int i);

        void onPositionUI(int i);

        void onReversalRotation(int i);

        void onSmoothUI(int i);

        void onSpeedModeUI(int i);

        void onSpeedUI(int i);

        void onStartPositionUI(int i);

        void onVibrateLinePoints(Map<String, Point> map);

        void onVibrateModeUI(int i);

        void onVibrateUI(int i);

        void onWebSpeedUI(int i);

        void onWebVibrateUI(int i);
    }

    public RemoteDataManager(Context context) {
        this.limitMax = 100;
        this.limitVibrateMax = 100;
        this.mContext = context;
        this.limitMax = ((Integer) SharedPreferencesUtil.get(context, Constans.SAFE_FUN1, 40)).intValue();
        this.limitVibrateMax = ((Integer) SharedPreferencesUtil.get(context, Constans.SAFE_FUN2, 40)).intValue();
    }

    public static RemoteDataManager getInstance() {
        RemoteDataManager remoteDataManager;
        synchronized (RemoteDataManager.class) {
            remoteDataManager = instance;
            if (remoteDataManager == null) {
                remoteDataManager = new RemoteDataManager(MyApplication.getApplication());
                instance = remoteDataManager;
            }
        }
        return remoteDataManager;
    }

    /* JADX WARN: Failed to restore switch over string. Please report as a decompilation issue */
    private void initBleAAData(String str, String str2) {
        try {
            int intValue = Integer.valueOf(str2, 16).intValue();
            switch (str.hashCode()) {
                case 1538:
                    if (str.equals("02")) {
                        this.mHandler.sendEmptyMessage(16);
                        break;
                    }
                    break;
                case 1539:
                    if (str.equals("03")) {
                        SocketManager.getInstance().setSpeed(intValue, 1);
                        Message message = new Message();
                        message.what = 1;
                        message.arg1 = intValue;
                        this.mHandler.sendMessage(message);
                        break;
                    }
                    break;
                case 1540:
                    if (str.equals("04")) {
                        if (intValue > 100) {
                            intValue = 100;
                        } else if (intValue < 0) {
                            intValue = 0;
                        }
                        SocketManager.getInstance().setSpeed(intValue, 1);
                        Message message2 = new Message();
                        message2.what = 1;
                        message2.arg1 = intValue;
                        this.mHandler.sendMessage(message2);
                        break;
                    }
                    break;
                case 1541:
                    if (str.equals("05")) {
                        SocketManager.getInstance().setMode(intValue, 1);
                        Message message3 = new Message();
                        message3.what = 2;
                        message3.arg1 = intValue;
                        this.mHandler.sendMessage(message3);
                        break;
                    }
                    break;
                case 1542:
                    if (str.equals("06") && MyApplication.getApplication().getmProductModel() != null) {
                        if (MyApplication.getApplication().getmProductModel().getClimax() != 1) {
                            SocketManager.getInstance().setVibrateMode(intValue, 1);
                            Message message4 = new Message();
                            message4.what = 4;
                            message4.arg1 = intValue;
                            this.mHandler.sendMessage(message4);
                            break;
                        } else {
                            this.mHandler.sendEmptyMessage(7);
                            break;
                        }
                    }
                    break;
                case 1543:
                    if (str.equals("07") && MyApplication.getApplication().getmProductModel() != null && MyApplication.getApplication().getmProductModel().getMusic() == 1) {
                        Message message5 = new Message();
                        message5.what = 17;
                        message5.arg1 = intValue;
                        this.mHandler.sendMessage(message5);
                        break;
                    }
                    break;
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /* JADX WARN: Failed to restore switch over string. Please report as a decompilation issue */
    private void initBleCCData(String str, String str2, String str3) {
        try {
            int parseInt = Integer.parseInt(str3, 16);
            int hashCode = str2.hashCode();
            if (hashCode == 1553) {
                if (str2.equals("0A")) {
                    SocketManager.getInstance().setStrokeNow(parseInt, 1);
                    Message message = new Message();
                    message.what = 14;
                    message.arg1 = parseInt;
                    this.mHandler.sendMessage(message);
                }
                return;
            }
            if (hashCode == 1554) {
                if (str2.equals(Constans.piuCommand)) {
                    int intValue = Integer.valueOf(str3, 16).intValue();
                    Message message2 = new Message();
                    message2.what = 24;
                    message2.obj = Integer.valueOf(intValue);
                    this.mHandler.sendMessage(message2);
                    return;
                }
                return;
            }
            switch (hashCode) {
                case 1537:
                    if (str2.equals(ModeDataVCUtil.HEAD_VERSION)) {
                        if (parseInt != 160) {
                            if (parseInt != 161) {
                                if (parseInt != 176) {
                                    if (parseInt != 177) {
                                        if (parseInt != 192) {
                                            if (parseInt == 193) {
                                                SocketManager.getInstance().setReversal(1, 1);
                                                Message message3 = new Message();
                                                message3.what = 23;
                                                message3.arg1 = 1;
                                                this.mHandler.sendMessage(message3);
                                                break;
                                            }
                                        } else {
                                            SocketManager.getInstance().setReversal(0, 1);
                                            Message message4 = new Message();
                                            message4.what = 23;
                                            message4.arg1 = 0;
                                            this.mHandler.sendMessage(message4);
                                            break;
                                        }
                                    } else {
                                        Message message5 = new Message();
                                        message5.what = 17;
                                        message5.arg1 = 1;
                                        this.mHandler.sendMessage(message5);
                                        break;
                                    }
                                } else {
                                    Message message6 = new Message();
                                    message6.what = 17;
                                    message6.arg1 = 0;
                                    this.mHandler.sendMessage(message6);
                                    break;
                                }
                            } else {
                                SocketManager.getInstance().setStrokeType(1, 1);
                                Message message7 = new Message();
                                message7.what = 18;
                                message7.arg1 = 1;
                                this.mHandler.sendMessage(message7);
                                break;
                            }
                        } else {
                            SocketManager.getInstance().setStrokeType(0, 1);
                            Message message8 = new Message();
                            message8.what = 18;
                            message8.arg1 = 0;
                            this.mHandler.sendMessage(message8);
                            break;
                        }
                    }
                    break;
                case 1538:
                    if (str2.equals("02")) {
                        SocketManager.getInstance().setStrokeType(0, 1);
                        this.mHandler.sendEmptyMessage(16);
                        break;
                    }
                    break;
                case 1539:
                    if (str2.equals("03")) {
                        if (parseInt > 100) {
                            parseInt = 100;
                        } else if (parseInt < 0) {
                            parseInt = 0;
                        }
                        SocketManager.getInstance().setSpeed(parseInt, 1);
                        Message message9 = new Message();
                        message9.what = 1;
                        message9.arg1 = parseInt;
                        this.mHandler.sendMessage(message9);
                        break;
                    }
                    break;
                case 1540:
                    if (str2.equals("04")) {
                        SocketManager.getInstance().setMode(parseInt, 1);
                        Message message10 = new Message();
                        message10.what = 2;
                        message10.arg1 = parseInt;
                        this.mHandler.sendMessage(message10);
                        break;
                    }
                    break;
                case 1541:
                    if (str2.equals("05")) {
                        SocketManager.getInstance().setVibrate(parseInt, 1);
                        Message message11 = new Message();
                        message11.what = 3;
                        message11.arg1 = parseInt;
                        this.mHandler.sendMessage(message11);
                        break;
                    }
                    break;
                case 1542:
                    if (str2.equals("06")) {
                        SocketManager.getInstance().setVibrateMode(parseInt, 1);
                        Message message12 = new Message();
                        message12.what = 4;
                        message12.arg1 = parseInt;
                        this.mHandler.sendMessage(message12);
                        break;
                    }
                    break;
                case 1543:
                    if (str2.equals("07")) {
                        Message message13 = new Message();
                        message13.what = 11;
                        message13.arg1 = parseInt;
                        this.mHandler.sendMessage(message13);
                        break;
                    }
                    break;
                case 1544:
                    if (str2.equals("08")) {
                        Message message14 = new Message();
                        message14.what = 12;
                        message14.arg1 = parseInt;
                        this.mHandler.sendMessage(message14);
                        break;
                    }
                    break;
                case 1545:
                    if (str2.equals("09")) {
                        SocketManager.getInstance().setSmooth(parseInt, 1);
                        Message message15 = new Message();
                        message15.what = 13;
                        message15.arg1 = parseInt;
                        this.mHandler.sendMessage(message15);
                        break;
                    }
                    break;
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void launchActivity(Context context) {
        Intent intent = new Intent("android.intent.action.MAIN");
        intent.addCategory("android.intent.category.LAUNCHER");
        intent.setFlags(270532608);
        intent.setComponent(new ComponentName(this.mContext.getPackageName(), "com.hismith.hismithapp.RemotePlayActivity"));
        context.startActivity(intent);
    }

    @Override // com.hismith.hismithapp.http.SocketManager.OnSocketNotifyListener
    public void OnNotify(String str) {
        if (TextUtils.isEmpty(str)) {
            return;
        }
        int i = 15;
        this.mHandler.sendEmptyMessage(15);
        try {
            RemotePlayModel remotePlayModel = (RemotePlayModel) this.gson.fromJson(str, RemotePlayModel.class);
            if (remotePlayModel.getSuccess() == 1) {
                if ("1".equals(remotePlayModel.getClientConnection())) {
                    this.mHandler.sendEmptyMessage(5);
                    return;
                }
                if ("1".equals(remotePlayModel.getOffline())) {
                    this.mHandler.sendEmptyMessage(6);
                    return;
                }
                int i2 = 0;
                if ("setSpeed".equals(remotePlayModel.getAction())) {
                    if (remotePlayModel.isMotor1()) {
                        int speed = remotePlayModel.getSpeed();
                        if (speed >= 0 && speed <= (i2 = this.limitMax)) {
                            i2 = speed;
                        }
                        Message message = new Message();
                        message.what = 1;
                        message.arg1 = i2;
                        this.mHandler.sendMessage(message);
                        Message message2 = new Message();
                        message2.what = 21;
                        message2.arg1 = remotePlayModel.getSpeed();
                        this.mHandler.sendMessage(message2);
                        BleManager.getInstance(this.mContext).writeSpeed(i2);
                        if (remotePlayModel.getLines() == null || !remotePlayModel.getLines().isJsonObject()) {
                            return;
                        }
                        JsonObject jsonObject = (JsonObject) remotePlayModel.getLines();
                        HashMap hashMap = new HashMap();
                        for (String str2 : jsonObject.keySet()) {
                            JsonObject asJsonObject = jsonObject.getAsJsonObject(str2);
                            int asInt = asJsonObject.get("x").getAsInt();
                            int asInt2 = asJsonObject.get("y").getAsInt();
                            if (asInt >= 0 && asInt2 >= 0) {
                                hashMap.put(str2, new Point(asInt, asInt2));
                                LogUtil.e("WebSocket line key:" + str2 + ", value:" + asJsonObject);
                            }
                        }
                        Message message3 = new Message();
                        message3.what = 19;
                        message3.obj = hashMap;
                        this.mHandler.sendMessage(message3);
                        return;
                    }
                    if (remotePlayModel.isMotor2()) {
                        int speed2 = remotePlayModel.getSpeed();
                        if (speed2 >= 0 && speed2 <= (i2 = this.limitVibrateMax)) {
                            i2 = speed2;
                        }
                        Message message4 = new Message();
                        message4.what = 3;
                        message4.arg1 = i2;
                        this.mHandler.sendMessage(message4);
                        Message message5 = new Message();
                        message5.what = 22;
                        message5.arg1 = remotePlayModel.getSpeed();
                        this.mHandler.sendMessage(message5);
                        BleManager.getInstance(this.mContext).writeVibrate(i2);
                        if (remotePlayModel.getLines() == null || !remotePlayModel.getLines().isJsonObject()) {
                            return;
                        }
                        JsonObject jsonObject2 = (JsonObject) remotePlayModel.getLines();
                        HashMap hashMap2 = new HashMap();
                        for (String str3 : jsonObject2.keySet()) {
                            JsonObject asJsonObject2 = jsonObject2.getAsJsonObject(str3);
                            int asInt3 = asJsonObject2.get("x").getAsInt();
                            int asInt4 = asJsonObject2.get("y").getAsInt();
                            if (asInt3 >= 0 && asInt4 >= 0) {
                                hashMap2.put(str3, new Point(asInt3, asInt4));
                                LogUtil.e("WebSocket line key:" + str3 + ", value:" + asJsonObject2);
                            }
                        }
                        Message message6 = new Message();
                        message6.what = 20;
                        message6.obj = hashMap2;
                        this.mHandler.sendMessage(message6);
                        return;
                    }
                    return;
                }
                if ("setStrokeNow".equals(remotePlayModel.getAction())) {
                    BleManager.getInstance(this.mContext).writePosition(remotePlayModel.getStrokeNow());
                    return;
                }
                if (!"setMode".equals(remotePlayModel.getAction())) {
                    if ("setClimax".equals(remotePlayModel.getAction())) {
                        if (MyApplication.getApplication().getmProductModel() == null || MyApplication.getApplication().getmProductModel().getClimax() != 1) {
                            return;
                        }
                        boolean equals = "1003".equals(MyApplication.getApplication().getDeviceCode());
                        Context context = this.mContext;
                        if (equals) {
                            BleManager.getInstance(context).writeOpenClimax();
                            return;
                        } else {
                            BleManager.getInstance(context).writeOpenClimax01();
                            return;
                        }
                    }
                    if ("setReversal".equals(remotePlayModel.getAction())) {
                        int reversal = remotePlayModel.getReversal();
                        if (reversal == 1) {
                            BleManager.getInstance(this.mContext).writeOpenReversal();
                            return;
                        } else {
                            if (reversal == 0) {
                                BleManager.getInstance(this.mContext).writeCloseReversal();
                                return;
                            }
                            return;
                        }
                    }
                    if ("setInject".equals(remotePlayModel.getAction())) {
                        BleManager.getInstance(this.mContext).writeOpenPiu();
                        return;
                    }
                    if ("chatMessage".equals(remotePlayModel.getAction())) {
                        if (TextUtils.isEmpty(remotePlayModel.getMessage())) {
                            return;
                        }
                        Message message7 = new Message();
                        message7.what = 25;
                        message7.obj = remotePlayModel.getMessage();
                        this.mHandler.sendMessage(message7);
                        return;
                    }
                    if ("runStroke".equals(remotePlayModel.getAction())) {
                        Integer strokeStart = remotePlayModel.getStrokeStart();
                        final Integer strokeEnd = remotePlayModel.getStrokeEnd();
                        if (strokeStart == null || strokeStart.intValue() < this.limitStartPosition) {
                            i = 0;
                        } else {
                            BleManager.getInstance(this.mContext).writeData(BleManager.getInstance(this.mContext).getBleCCData(7, strokeStart.intValue()));
                        }
                        if (strokeEnd == null || strokeEnd.intValue() > this.limitStopPosition) {
                            return;
                        }
                        this.mHandler.postDelayed(new Runnable() { // from class: com.hismith.hismithapp.manager.RemoteDataManager.3
                            @Override // java.lang.Runnable
                            public void run() {
                                BleManager.getInstance(RemoteDataManager.this.mContext).writeData(BleManager.getInstance(RemoteDataManager.this.mContext).getBleCCData(8, strokeEnd.intValue()));
                            }
                        }, i);
                        return;
                    }
                    return;
                }
                int mode = remotePlayModel.getMode();
                if (remotePlayModel.isMotor1Mode()) {
                    if (mode <= 0) {
                        Message message8 = new Message();
                        message8.what = 1;
                        message8.arg1 = this.mSpeed;
                        this.mHandler.sendMessage(message8);
                        BleManager.getInstance(this.mContext).writeSpeed(this.mSpeed);
                        return;
                    }
                    Message message9 = new Message();
                    message9.what = 2;
                    message9.arg1 = mode;
                    this.mHandler.sendMessage(message9);
                    boolean hasMode = ProductUtil.hasMode(0, mode);
                    Context context2 = this.mContext;
                    if (hasMode) {
                        BleManager.getInstance(context2).writeData(ProductUtil.getProductMode(0)[mode - 1].getmCommand());
                        return;
                    } else {
                        BleManager.getInstance(context2).writeSpeedMode(mode);
                        return;
                    }
                }
                if (remotePlayModel.isMotor2Mode()) {
                    if (mode != 0) {
                        Message message10 = new Message();
                        message10.what = 4;
                        message10.arg1 = mode;
                        this.mHandler.sendMessage(message10);
                        boolean hasMode2 = ProductUtil.hasMode(1, mode);
                        Context context3 = this.mContext;
                        if (hasMode2) {
                            BleManager.getInstance(context3).writeData(ProductUtil.getProductMode(1)[mode - 1].getmCommand());
                            return;
                        } else {
                            BleManager.getInstance(context3).writeVibrateMode(mode);
                            return;
                        }
                    }
                    if (this.mVibrate > 0) {
                        Message message11 = new Message();
                        message11.what = 3;
                        message11.arg1 = this.mVibrate;
                        this.mHandler.sendMessage(message11);
                        BleManager.getInstance(this.mContext).writeVibrate(this.mVibrate);
                        return;
                    }
                    Message message12 = new Message();
                    message12.what = 4;
                    message12.arg1 = 0;
                    this.mHandler.sendMessage(message12);
                    BleManager.getInstance(this.mContext).writeCloseVibrate();
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void destroyFloatWindow() {
        if (SocketManager.getInstance().getOnSocketNotifyListener() instanceof RemoteDataManager) {
            LogUtil.e("------------------清除回调------------------");
            SocketManager.getInstance().setOnSocketNotifyListener(null);
        }
        LogUtil.e("准备清除悬浮窗");
        if (this.mFloatBase != null) {
            LogUtil.e("清除悬浮窗");
            this.mFloatBase.destroy();
            this.mFloatBase = null;
        }
    }

    public int getLimitMax() {
        return this.limitMax;
    }

    public int getLimitStartPosition() {
        return this.limitStartPosition;
    }

    public int getLimitStopPosition() {
        return this.limitStopPosition;
    }

    public int getLimitVibrateMax() {
        return this.limitVibrateMax;
    }

    public void initListener() {
        BleManager.getInstance(this.mContext).setOnNotifyChangeListener_v2(this);
        SocketManager.getInstance().setOnSocketNotifyListener(this);
    }

    @Override // com.hismith.hismithapp.http.SocketManager.OnSocketNotifyListener
    public void onClosed(int i, String str) {
        this.mHandler.sendEmptyMessage(9);
    }

    @Override // com.hismith.hismithapp.http.SocketManager.OnSocketNotifyListener
    public void onCodeNull() {
        this.mHandler.sendEmptyMessage(10);
    }

    @Override // com.hismith.hismithapp.http.SocketManager.OnSocketNotifyListener
    public void onFailure(Throwable th, Response response) {
        this.mHandler.sendEmptyMessage(9);
    }

    @Override // com.hismith.hismithapp.bluetooth.BleManager.OnNotifyChangeListener_v2
    public void onNotifyChange_v2(String str, String str2, String str3) {
        if ("AA".equals(str)) {
            initBleAAData(str2, str3);
        } else if ("CA".equals(str) || "CC".equals(str)) {
            initBleCCData(str, str2, str3);
        }
    }

    @Override // com.hismith.hismithapp.http.SocketManager.OnSocketNotifyListener
    public void onOpen() {
        this.mHandler.sendEmptyMessage(8);
    }

    public void release() {
        if (BleManager.getInstance(this.mContext).getOnNotifyChangeListener_v2() instanceof RemoteDataManager) {
            BleManager.getInstance(this.mContext).setOnNotifyChangeListener_v2(null);
        }
        if (instance != null) {
            instance = null;
        }
    }

    public void setCallback(Callback callback) {
        this.callback = callback;
    }

    public void setLimitMax(int i) {
        this.limitMax = i;
        SharedPreferencesUtil.put(this.mContext, Constans.SAFE_FUN1, Integer.valueOf(i));
    }

    public void setLimitStartPosition(int i) {
        this.limitStartPosition = i;
    }

    public void setLimitStopPosition(int i) {
        this.limitStopPosition = i;
    }

    public void setLimitVibrateMax(int i) {
        this.limitVibrateMax = i;
        SharedPreferencesUtil.put(this.mContext, Constans.SAFE_FUN2, Integer.valueOf(i));
    }

    public void showFloatWindow() {
        FloatBase floatBase = this.mFloatBase;
        if (floatBase == null) {
            floatBase = FloatManager.getFloat(this.mContext, MyApplication.getApplication().getmProductModel() != null ? MyApplication.getApplication().getmProductModel().getDeviceType() : "", Constans.FLOAT_REMOTE_WINDOW, new View.OnClickListener() { // from class: com.hismith.hismithapp.manager.RemoteDataManager.2
                @Override // android.view.View.OnClickListener
                public void onClick(View view) {
                    LogUtil.e("BleManager 点击悬浮窗" + (!AppBackgroundUtil.isAppBackground()));
                    if (AppBackgroundUtil.isAppBackground()) {
                        return;
                    }
                    RemoteDataManager remoteDataManager = RemoteDataManager.this;
                    remoteDataManager.launchActivity(remoteDataManager.mContext);
                    RemoteDataManager.this.destroyFloatWindow();
                }
            });
            this.mFloatBase = floatBase;
        }
        if (floatBase != null) {
            floatBase.show();
        }
    }
}
