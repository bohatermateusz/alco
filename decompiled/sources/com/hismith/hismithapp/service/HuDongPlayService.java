package com.hismith.hismithapp.service;

import android.app.Service;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.os.Binder;
import android.os.Handler;
import android.os.IBinder;
import android.os.Message;
import android.text.TextUtils;
import android.view.View;
import androidx.exifinterface.media.ExifInterface;
import androidx.media3.exoplayer.ExoPlayer;
import com.google.gson.Gson;
import com.hismith.hismithapp.ControlAcceptActivity2;
import com.hismith.hismithapp.ControlSponsorActivity2;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.bluetooth.BleManager;
import com.hismith.hismithapp.db.DBMessageUtil;
import com.hismith.hismithapp.floating.FloatBase;
import com.hismith.hismithapp.floating.FloatManager;
import com.hismith.hismithapp.http.HttpModel;
import com.hismith.hismithapp.http.RetrofitRequest;
import com.hismith.hismithapp.http.RxSubscribe;
import com.hismith.hismithapp.http.SocketManager;
import com.hismith.hismithapp.manager.RemoteFloatingManager;
import com.hismith.hismithapp.model.SocketProductModel;
import com.hismith.hismithapp.model.SocketSpeedModel;
import com.hismith.hismithapp.observer.AppBackgroundUtil;
import com.hismith.hismithapp.product.ProductUtil;
import com.hismith.hismithapp.utils.AppManager;
import com.hismith.hismithapp.utils.AppUtil;
import com.hismith.hismithapp.utils.CacheUtil;
import com.hismith.hismithapp.utils.CheckUtil;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.IOUtils;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.ModeDataVCUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import com.hismith.hismithapp.utils.ToastUtil;
import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.observers.DisposableObserver;
import io.reactivex.schedulers.Schedulers;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import okhttp3.ResponseBody;

/* loaded from: classes3.dex */
public class HuDongPlayService extends Service {
    private static final int WHAT_NO_ANSWER = 1;
    private static int mClientConnection;
    private static int mServerConnection;
    private RetrofitRequest baseRequest;
    private HuDongBinder mBinder;
    private FloatBase mFloatBase;
    private Gson mGson;
    private OnServiceNotifyListener mOnServiceNotifyListener;
    private SocketProductModel mSocketProductModel;
    private String name = "";
    private String headIcon = "";
    private int type = 0;
    private int msgId = 0;
    private String gameToken = "";
    private int mSpeed = 0;
    private int mSafeMax = 100;
    private int mVibrateSafeMax = 100;
    private boolean isFirst = true;
    private int mStartPositionSafeMax = 0;
    private int mStopPositionSafeMax = 100;
    private boolean mServerSwitch = false;
    private boolean isShowing = false;
    private int TIME_NO_ANSWER = ExoPlayer.DEFAULT_STUCK_PLAYING_NOT_ENDING_TIMEOUT_MS;
    private Handler mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.service.HuDongPlayService.1
        @Override // android.os.Handler.Callback
        public boolean handleMessage(Message message) {
            if (message.what != 1) {
                return false;
            }
            HuDongPlayService huDongPlayService = HuDongPlayService.this;
            ToastUtil.showShortToast(huDongPlayService, LanguageUtil.getLanguageByKey(huDongPlayService, "otherNotReceive"));
            if (HuDongPlayService.this.mBinder == null) {
                return false;
            }
            HuDongPlayService.this.mBinder.exit();
            return false;
        }
    });
    private BroadcastReceiver mReceiver = new BroadcastReceiver() { // from class: com.hismith.hismithapp.service.HuDongPlayService.2
        @Override // android.content.BroadcastReceiver
        public void onReceive(Context context, Intent intent) {
            String action = intent.getAction();
            if (Constans.RECEIVER_PRODUCT_INFO.equals(action)) {
                if (HuDongPlayService.this.mBinder != null && CheckUtil.checkOp(HuDongPlayService.this, 24) == 0 && HuDongPlayService.this.isShowing) {
                    HuDongPlayService.this.mBinder.destroyFloatWindow();
                    HuDongPlayService.this.mBinder.showFloatWindow();
                    return;
                }
                return;
            }
            if (!Constans.RECEIVER_BLE_RECE_DATA.equals(action)) {
                if (SocketManager.RECEIVER_SOCKET_NOTIFY.equals(action)) {
                    HuDongPlayService.this.initSocketData(intent.getIntExtra(SocketManager.EXTRA_SOCKET_ERROR, -1), intent.getStringExtra(SocketManager.EXTRA_SOCKET_DATA));
                    return;
                }
                return;
            }
            String stringExtra = intent.getStringExtra(Constans.BleType);
            String stringExtra2 = intent.getStringExtra(Constans.BleData0);
            String stringExtra3 = intent.getStringExtra(Constans.BleData1);
            LogUtil.e("蓝牙广播数据接收：" + stringExtra + ", data0:" + stringExtra2 + ", data1:" + stringExtra3);
            if ("AA".equals(stringExtra)) {
                HuDongPlayService.this.initBleAAData(stringExtra2, stringExtra3);
            } else if ("CA".equals(stringExtra) || "CC".equals(stringExtra)) {
                HuDongPlayService.this.initBleCCData(stringExtra, stringExtra2, stringExtra3);
            }
        }
    };

    public class HuDongBinder extends Binder {
        public HuDongBinder() {
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void launchActivity() {
            Intent intent = HuDongPlayService.mClientConnection == 1 ? new Intent(HuDongPlayService.this, (Class<?>) ControlSponsorActivity2.class) : HuDongPlayService.mServerConnection == 1 ? new Intent(HuDongPlayService.this, (Class<?>) ControlAcceptActivity2.class) : null;
            if (intent != null) {
                intent.addFlags(268435456);
                HuDongPlayService.this.startActivity(intent);
            }
        }

        public void controlAccept() {
            LogUtil.e("controlAccept()");
            RemoteFloatingManager.getInstance().destroyFloatWindow();
            if (TextUtils.isEmpty(HuDongPlayService.this.gameToken)) {
                return;
            }
            SocketManager.getInstance().setGameToken(HuDongPlayService.this.gameToken);
            SocketManager.getInstance().connSocket(4);
        }

        public void destroyFloatWindow() {
            LogUtil.e("准备清除悬浮窗");
            HuDongPlayService.this.isShowing = false;
            if (HuDongPlayService.this.mFloatBase != null) {
                LogUtil.e("清除悬浮窗");
                HuDongPlayService.this.mFloatBase.destroy();
                HuDongPlayService.this.mFloatBase = null;
            }
        }

        public void exit() {
            LogUtil.e("Service exit()");
            AppManager.getInstance().finishActivity(ControlSponsorActivity2.class);
            AppManager.getInstance().finishActivity(ControlAcceptActivity2.class);
            HuDongPlayService.this.closeSocket();
            if (ProductUtil.isSMSType()) {
                int i = 15;
                int i2 = 0;
                if (HuDongPlayService.this.mStartPositionSafeMax < BleManager.getInstance(HuDongPlayService.this).getmConnBleDevice().getStopPosition()) {
                    i2 = 15;
                    i = 0;
                }
                new Handler().postDelayed(new Runnable() { // from class: com.hismith.hismithapp.service.HuDongPlayService.HuDongBinder.3
                    @Override // java.lang.Runnable
                    public void run() {
                        BleManager.getInstance(HuDongPlayService.this).writeData(BleManager.getInstance(HuDongPlayService.this).getBleCCData(7, HuDongPlayService.this.mStartPositionSafeMax));
                    }
                }, i);
                new Handler().postDelayed(new Runnable() { // from class: com.hismith.hismithapp.service.HuDongPlayService.HuDongBinder.4
                    @Override // java.lang.Runnable
                    public void run() {
                        BleManager.getInstance(HuDongPlayService.this).writeData(BleManager.getInstance(HuDongPlayService.this).getBleCCData(8, HuDongPlayService.this.mStopPositionSafeMax));
                    }
                }, i2);
            }
            new Handler().postDelayed(new Runnable() { // from class: com.hismith.hismithapp.service.HuDongPlayService.HuDongBinder.5
                @Override // java.lang.Runnable
                public void run() {
                    BleManager.getInstance(HuDongPlayService.this).writeShutdown();
                }
            }, 100L);
            destroyFloatWindow();
            HuDongPlayService.this.mHandler.removeCallbacksAndMessages(null);
            HuDongPlayService.this.stopSelf();
        }

        public String getGameToken() {
            return HuDongPlayService.this.gameToken;
        }

        public String getHeadIcon() {
            return HuDongPlayService.this.headIcon;
        }

        public int getMsgId() {
            return HuDongPlayService.this.msgId;
        }

        public String getName() {
            return HuDongPlayService.this.name;
        }

        public int getSafeMax() {
            return HuDongPlayService.this.mSafeMax;
        }

        public SocketProductModel getSocketProductModel() {
            return HuDongPlayService.this.mSocketProductModel;
        }

        public int getVibrateSafeMax() {
            return HuDongPlayService.this.mVibrateSafeMax;
        }

        public int getmStartPositionSafeMax() {
            return HuDongPlayService.this.mStartPositionSafeMax;
        }

        public int getmStopPositionSafeMax() {
            return HuDongPlayService.this.mStopPositionSafeMax;
        }

        public boolean isServerSwitch() {
            return HuDongPlayService.this.mServerSwitch;
        }

        public void msgAction(int i, int i2) {
            String str;
            String str2 = (String) SharedPreferencesUtil.get(HuDongPlayService.this, Constans.Token);
            if (i2 == 1) {
                str = "delete";
            } else if (i2 == 2) {
                str = "revoke";
            } else if (i2 == 3) {
                str = "revoked";
            } else if (i2 == 4) {
                str = "rejectRP";
            } else if (i2 != 5) {
                return;
            } else {
                str = "rejected";
            }
            HuDongPlayService.this.baseRequest.getApi().msgAction(str2, i, str).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<HttpModel>(HuDongPlayService.this, null, false) { // from class: com.hismith.hismithapp.service.HuDongPlayService.HuDongBinder.1
                @Override // com.hismith.hismithapp.http.RxSubscribe
                protected void onSuccess(HttpModel httpModel) {
                    httpModel.getSuccess();
                }
            });
        }

        public void setGameToken(String str) {
            HuDongPlayService.this.gameToken = str;
        }

        public void setHeadIcon(String str) {
            HuDongPlayService.this.headIcon = str;
        }

        public void setMsgId(int i) {
            HuDongPlayService.this.msgId = i;
        }

        public void setName(String str) {
            HuDongPlayService.this.name = str;
        }

        public void setOnServiceNotifyListener(OnServiceNotifyListener onServiceNotifyListener) {
            HuDongPlayService.this.mOnServiceNotifyListener = onServiceNotifyListener;
        }

        public void setSafeMax(int i) {
            HuDongPlayService.this.mSafeMax = i;
            HuDongPlayService huDongPlayService = HuDongPlayService.this;
            SharedPreferencesUtil.put(huDongPlayService, Constans.SAFE_FUN1, Integer.valueOf(huDongPlayService.mSafeMax));
        }

        public void setServerSwitch(boolean z) {
            HuDongPlayService.this.mServerSwitch = z;
        }

        public void setSocketProductModel(SocketProductModel socketProductModel) {
            HuDongPlayService.this.mSocketProductModel = socketProductModel;
        }

        public void setVibrateSafeMax(int i) {
            HuDongPlayService.this.mVibrateSafeMax = i;
            HuDongPlayService huDongPlayService = HuDongPlayService.this;
            SharedPreferencesUtil.put(huDongPlayService, Constans.SAFE_FUN2, Integer.valueOf(huDongPlayService.mVibrateSafeMax));
        }

        public void setmStartPositionSafeMax(int i) {
            HuDongPlayService.this.mStartPositionSafeMax = i;
        }

        public void setmStopPositionSafeMax(int i) {
            HuDongPlayService.this.mStopPositionSafeMax = i;
        }

        public void showFloatWindow() {
            HuDongPlayService.this.isShowing = true;
            if (HuDongPlayService.this.mFloatBase == null) {
                String deviceType = MyApplication.getApplication().getmProductModel() != null ? MyApplication.getApplication().getmProductModel().getDeviceType() : "";
                HuDongPlayService huDongPlayService = HuDongPlayService.this;
                huDongPlayService.mFloatBase = FloatManager.getFloat(huDongPlayService, deviceType, Constans.FLOAT_HU_DONG_WINDOW, new View.OnClickListener() { // from class: com.hismith.hismithapp.service.HuDongPlayService.HuDongBinder.2
                    @Override // android.view.View.OnClickListener
                    public void onClick(View view) {
                        LogUtil.e("BleManager 点击悬浮窗");
                        if (AppBackgroundUtil.isAppBackground()) {
                            return;
                        }
                        HuDongBinder.this.launchActivity();
                        HuDongBinder.this.destroyFloatWindow();
                    }
                });
            }
            HuDongPlayService.this.initBleFloatData();
            if (HuDongPlayService.this.mFloatBase != null) {
                HuDongPlayService.this.mFloatBase.show();
            }
        }
    }

    public interface OnServiceNotifyListener {
        void onProductChange(SocketProductModel socketProductModel);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void closeSocket() {
        if (SocketManager.getInstance().getWebSocket() != null) {
            LogUtil.e("WebSocket 发送：关闭");
            SocketManager.getInstance().close();
            SocketManager.getInstance().closeSocket();
        }
    }

    private void downloadSVGFile(final String str, final String str2, final boolean z) {
        LogUtil.e("SVG文件下载函数" + str2);
        this.baseRequest.getApi().download(str2).subscribeOn(Schedulers.io()).observeOn(Schedulers.newThread()).subscribe(new DisposableObserver<ResponseBody>() { // from class: com.hismith.hismithapp.service.HuDongPlayService.5
            @Override // io.reactivex.Observer
            public void onComplete() {
                LogUtil.e("SVG下载完成");
                if (z) {
                    HuDongPlayService.this.sendBroadcast(new Intent(Constans.RECEIVER_SVG_DOWNLOAD));
                }
            }

            @Override // io.reactivex.Observer
            public void onError(Throwable th) {
                LogUtil.e("SVG下载响应错误");
            }

            @Override // io.reactivex.Observer
            public void onNext(ResponseBody responseBody) {
                InputStream inputStream;
                FileOutputStream fileOutputStream;
                LogUtil.e("SVG下载响应成功:" + str2);
                FileOutputStream fileOutputStream2 = null;
                try {
                    inputStream = responseBody.byteStream();
                    try {
                        try {
                            File file = new File(MyApplication.getApplication().getFilesDir() + "/playSVG");
                            if (!file.exists()) {
                                file.mkdir();
                            }
                            File file2 = new File(file, str);
                            if (!file2.exists()) {
                                file2.createNewFile();
                            }
                            fileOutputStream = new FileOutputStream(file2);
                        } catch (Exception e) {
                            e = e;
                        }
                    } catch (Throwable th) {
                        th = th;
                    }
                } catch (Exception e2) {
                    e = e2;
                    inputStream = null;
                } catch (Throwable th2) {
                    th = th2;
                    inputStream = null;
                }
                try {
                    byte[] bArr = new byte[2048];
                    while (true) {
                        int read = inputStream.read(bArr);
                        if (read == -1) {
                            break;
                        } else {
                            fileOutputStream.write(bArr, 0, read);
                        }
                    }
                    fileOutputStream.flush();
                    IOUtils.closeQuietly(fileOutputStream);
                } catch (Exception e3) {
                    fileOutputStream2 = fileOutputStream;
                    e = e3;
                    e.printStackTrace();
                    IOUtils.closeQuietly(fileOutputStream2);
                    IOUtils.closeQuietly(inputStream);
                } catch (Throwable th3) {
                    fileOutputStream2 = fileOutputStream;
                    th = th3;
                    IOUtils.closeQuietly(fileOutputStream2);
                    IOUtils.closeQuietly(inputStream);
                    throw th;
                }
                IOUtils.closeQuietly(inputStream);
            }
        });
    }

    public static int getClientConnection() {
        return mClientConnection;
    }

    public static int getServerConnection() {
        return mServerConnection;
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void initBleAAData(String str, String str2) {
        FloatBase floatBase;
        FloatBase floatBase2;
        try {
            int parseInt = Integer.parseInt(str2, 16);
            switch (str.hashCode()) {
                case 1538:
                    if (!str.equals("02") || (floatBase = this.mFloatBase) == null) {
                        return;
                    }
                    floatBase.refreshFloatShutdown();
                    return;
                case 1539:
                    if (!str.equals("03")) {
                        return;
                    }
                    break;
                case 1540:
                    if (str.equals("04")) {
                        break;
                    } else {
                        return;
                    }
                case 1541:
                    if (str.equals("05")) {
                        SocketManager.getInstance().setMode(parseInt, 1);
                        FloatBase floatBase3 = this.mFloatBase;
                        if (floatBase3 != null) {
                            floatBase3.refreshFloatModeUI(parseInt);
                            return;
                        }
                        return;
                    }
                    return;
                case 1542:
                    if (!str.equals("06") || MyApplication.getApplication().getmProductModel() == null) {
                        return;
                    }
                    if (MyApplication.getApplication().getmProductModel().getClimax() == 1) {
                        FloatBase floatBase4 = this.mFloatBase;
                        if (floatBase4 != null) {
                            floatBase4.refreshFloatClimaxUI(parseInt);
                            return;
                        }
                        return;
                    }
                    if (Constans.DEVICE_TYPE_SM2.equals(MyApplication.getApplication().getmProductModel().getDeviceType())) {
                        SocketManager.getInstance().setVibrateMode(parseInt, 1);
                        FloatBase floatBase5 = this.mFloatBase;
                        if (floatBase5 != null) {
                            floatBase5.refreshFloatVibrateModeUI(parseInt);
                            return;
                        }
                        return;
                    }
                    return;
                case 1543:
                    if (!str.equals("07") || MyApplication.getApplication().getmProductModel() == null || MyApplication.getApplication().getmProductModel().getMusic() != 1 || (floatBase2 = this.mFloatBase) == null) {
                        return;
                    }
                    floatBase2.refreshFloatMusicUI(parseInt);
                    return;
                default:
                    return;
            }
            SocketManager.getInstance().setSpeed(parseInt, 1);
            FloatBase floatBase6 = this.mFloatBase;
            if (floatBase6 != null) {
                floatBase6.refreshFloatSpeedUI(parseInt);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    /* JADX WARN: Failed to restore switch over string. Please report as a decompilation issue */
    public void initBleCCData(String str, String str2, String str3) {
        FloatBase floatBase;
        FloatBase floatBase2;
        FloatBase floatBase3;
        FloatBase floatBase4;
        try {
            int parseInt = Integer.parseInt(str3, 16);
            int hashCode = str2.hashCode();
            if (hashCode == 1553) {
                if (str2.equals("0A")) {
                    SocketManager.getInstance().setStrokeNow(parseInt, 1);
                    FloatBase floatBase5 = this.mFloatBase;
                    if (floatBase5 != null) {
                        floatBase5.refreshFloatPosition(parseInt);
                    }
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
                                                FloatBase floatBase6 = this.mFloatBase;
                                                if (floatBase6 != null) {
                                                    floatBase6.refreshFloatReversal(1);
                                                    break;
                                                }
                                            }
                                        } else {
                                            SocketManager.getInstance().setReversal(0, 1);
                                            FloatBase floatBase7 = this.mFloatBase;
                                            if (floatBase7 != null) {
                                                floatBase7.refreshFloatReversal(0);
                                                break;
                                            }
                                        }
                                    } else {
                                        FloatBase floatBase8 = this.mFloatBase;
                                        if (floatBase8 != null) {
                                            floatBase8.refreshFloatMusicUI(1);
                                            break;
                                        }
                                    }
                                } else {
                                    FloatBase floatBase9 = this.mFloatBase;
                                    if (floatBase9 != null) {
                                        floatBase9.refreshFloatMusicUI(0);
                                        break;
                                    }
                                }
                            } else {
                                SocketManager.getInstance().setStrokeType(1, 1);
                                FloatBase floatBase10 = this.mFloatBase;
                                if (floatBase10 != null) {
                                    floatBase10.refreshFloatRunMode(1);
                                    break;
                                }
                            }
                        } else {
                            SocketManager.getInstance().setStrokeType(0, 1);
                            FloatBase floatBase11 = this.mFloatBase;
                            if (floatBase11 != null) {
                                floatBase11.refreshFloatRunMode(0);
                                break;
                            }
                        }
                    }
                    break;
                case 1538:
                    if (str2.equals("02") && (floatBase = this.mFloatBase) != null) {
                        floatBase.refreshFloatShutdown();
                        break;
                    }
                    break;
                case 1539:
                    if (str2.equals("03")) {
                        SocketManager.getInstance().setSpeed(parseInt, 1);
                        FloatBase floatBase12 = this.mFloatBase;
                        if (floatBase12 != null) {
                            floatBase12.refreshFloatSpeedUI(parseInt);
                            break;
                        }
                    }
                    break;
                case 1540:
                    if (str2.equals("04")) {
                        SocketManager.getInstance().setMode(parseInt, 1);
                        FloatBase floatBase13 = this.mFloatBase;
                        if (floatBase13 != null) {
                            floatBase13.refreshFloatModeUI(parseInt);
                            break;
                        }
                    }
                    break;
                case 1541:
                    if (str2.equals("05")) {
                        SocketManager.getInstance().setVibrate(parseInt, 1);
                        FloatBase floatBase14 = this.mFloatBase;
                        if (floatBase14 != null) {
                            floatBase14.refreshFloatVibrateUI(parseInt);
                            break;
                        }
                    }
                    break;
                case 1542:
                    if (str2.equals("06")) {
                        SocketManager.getInstance().setVibrateMode(parseInt, 1);
                        FloatBase floatBase15 = this.mFloatBase;
                        if (floatBase15 != null) {
                            floatBase15.refreshFloatVibrateModeUI(parseInt);
                            break;
                        }
                    }
                    break;
                case 1543:
                    if (str2.equals("07") && (floatBase2 = this.mFloatBase) != null) {
                        floatBase2.refreshFloatStartPosition(parseInt);
                        break;
                    }
                    break;
                case 1544:
                    if (str2.equals("08") && (floatBase3 = this.mFloatBase) != null) {
                        floatBase3.refreshFloatStopPosition(parseInt);
                        break;
                    }
                    break;
                case 1545:
                    if (str2.equals("09") && (floatBase4 = this.mFloatBase) != null) {
                        floatBase4.refreshFloatSmooth(parseInt);
                        break;
                    }
                    break;
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void initBleFloatData() {
        FloatBase floatBase;
        if (BleManager.getInstance(this).getmConnBleDevice() != null) {
            int runMode = BleManager.getInstance(this).getmConnBleDevice().getRunMode();
            FloatBase floatBase2 = this.mFloatBase;
            if (runMode == 0) {
                if (floatBase2 != null) {
                    floatBase2.refreshFloatRunMode(0);
                }
            } else if (floatBase2 != null) {
                floatBase2.refreshFloatRunMode(1);
            }
            FloatBase floatBase3 = this.mFloatBase;
            if (floatBase3 != null) {
                floatBase3.refreshFloatSpeedUI(BleManager.getInstance(this).getmConnBleDevice().getSpeed());
            }
            FloatBase floatBase4 = this.mFloatBase;
            if (floatBase4 != null) {
                floatBase4.refreshFloatModeUI(BleManager.getInstance(this).getmConnBleDevice().getMode());
            }
            FloatBase floatBase5 = this.mFloatBase;
            if (floatBase5 != null) {
                floatBase5.refreshFloatVibrateUI(BleManager.getInstance(this).getmConnBleDevice().getVibration());
            }
            FloatBase floatBase6 = this.mFloatBase;
            if (floatBase6 != null) {
                floatBase6.refreshFloatVibrateModeUI(BleManager.getInstance(this).getmConnBleDevice().getVibrationMode());
            }
            int startPosition = BleManager.getInstance(this).getmConnBleDevice().getStartPosition();
            FloatBase floatBase7 = this.mFloatBase;
            if (floatBase7 != null) {
                floatBase7.refreshFloatStartPosition(startPosition);
            }
            int stopPosition = BleManager.getInstance(this).getmConnBleDevice().getStopPosition();
            FloatBase floatBase8 = this.mFloatBase;
            if (floatBase8 != null) {
                if (stopPosition <= 0) {
                    stopPosition = Constans.start_stop_position_diff + startPosition;
                }
                floatBase8.refreshFloatStartPosition(stopPosition);
            }
            FloatBase floatBase9 = this.mFloatBase;
            if (floatBase9 != null) {
                floatBase9.refreshFloatSmooth(BleManager.getInstance(this).getmConnBleDevice().getSmoothness());
            }
            FloatBase floatBase10 = this.mFloatBase;
            if (floatBase10 != null) {
                floatBase10.refreshFloatPosition(BleManager.getInstance(this).getmConnBleDevice().getPosition());
            }
            if (MyApplication.getApplication().getmProductModel() == null || MyApplication.getApplication().getmProductModel().getMusic() != 1 || (floatBase = this.mFloatBase) == null) {
                return;
            }
            floatBase.refreshFloatMusicUI(BleManager.getInstance(this).getmConnBleDevice().getMusic());
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void initSocketData(int i, String str) {
        int i2 = 0;
        try {
            if (i != 0) {
                mClientConnection = 0;
                mServerConnection = 0;
                HuDongBinder huDongBinder = this.mBinder;
                if (huDongBinder != null) {
                    huDongBinder.exit();
                    return;
                }
                return;
            }
            final SocketSpeedModel socketSpeedModel = (SocketSpeedModel) this.mGson.fromJson(str, SocketSpeedModel.class);
            if (socketSpeedModel == null || socketSpeedModel.getSuccess() != 1) {
                return;
            }
            String action = socketSpeedModel.getAction();
            if (!TextUtils.isEmpty(socketSpeedModel.getGameToken())) {
                LogUtil.e("连接成功，正在通知对方。。。");
                return;
            }
            if (socketSpeedModel.getClientConnection() == 1) {
                if (this.isFirst) {
                    this.mStartPositionSafeMax = BleManager.getInstance(this).getmConnBleDevice().getStartPosition();
                    this.mStopPositionSafeMax = BleManager.getInstance(this).getmConnBleDevice().getStopPosition();
                    this.isFirst = false;
                }
                DBMessageUtil.updateDBReject(this, this.msgId + "", ExifInterface.GPS_MEASUREMENT_3D);
                SocketManager.getInstance().setMsgId(this.msgId);
                SocketManager.getInstance().setRemote_time(System.currentTimeMillis());
                mClientConnection = 1;
                this.mServerSwitch = true;
                this.mHandler.removeMessages(1);
                SocketProductModel product = socketSpeedModel.getProduct();
                this.mSocketProductModel = product;
                presenterSVGFile(product);
                OnServiceNotifyListener onServiceNotifyListener = this.mOnServiceNotifyListener;
                if (onServiceNotifyListener != null) {
                    onServiceNotifyListener.onProductChange(this.mSocketProductModel);
                    return;
                }
                return;
            }
            if (socketSpeedModel.getServerConnection() == 1) {
                if (this.isFirst) {
                    this.mStartPositionSafeMax = BleManager.getInstance(this).getmConnBleDevice().getStartPosition();
                    this.mStopPositionSafeMax = BleManager.getInstance(this).getmConnBleDevice().getStopPosition();
                    this.isFirst = false;
                }
                DBMessageUtil.updateDBReject(this, this.msgId + "", ExifInterface.GPS_MEASUREMENT_3D);
                SocketManager.getInstance().setMsgId(this.msgId);
                SocketManager.getInstance().setRemote_time(System.currentTimeMillis());
                mServerConnection = 1;
                this.mHandler.removeMessages(1);
                SocketProductModel product2 = socketSpeedModel.getProduct();
                this.mSocketProductModel = product2;
                presenterSVGFile(product2);
                OnServiceNotifyListener onServiceNotifyListener2 = this.mOnServiceNotifyListener;
                if (onServiceNotifyListener2 != null) {
                    onServiceNotifyListener2.onProductChange(this.mSocketProductModel);
                    return;
                }
                return;
            }
            if (socketSpeedModel.getOffline() == 1) {
                ToastUtil.showShortToast(this, LanguageUtil.getLanguageByKey(this, "otherDisconnected"));
                mClientConnection = 0;
                mServerConnection = 0;
                HuDongBinder huDongBinder2 = this.mBinder;
                if (huDongBinder2 != null) {
                    huDongBinder2.exit();
                    return;
                }
                return;
            }
            if ("setSpeed".equals(action)) {
                if (socketSpeedModel.isMotor1()) {
                    int speed = socketSpeedModel.getSpeed();
                    this.mSpeed = speed;
                    int i3 = this.mSafeMax;
                    if (speed > i3) {
                        speed = i3;
                    }
                    if (socketSpeedModel.getOnlyRead() == 0 && this.mServerSwitch) {
                        BleManager.getInstance(this).writeSpeed(speed);
                        return;
                    } else {
                        if (socketSpeedModel.getOnlyRead() == 1) {
                            this.mSocketProductModel.setSpeed(speed);
                            return;
                        }
                        return;
                    }
                }
                if (socketSpeedModel.isMotor2()) {
                    int speed2 = socketSpeedModel.getSpeed();
                    int i4 = this.mVibrateSafeMax;
                    if (speed2 > i4) {
                        speed2 = i4;
                    }
                    if (socketSpeedModel.getOnlyRead() == 0 && this.mServerSwitch) {
                        BleManager.getInstance(this).writeVibrate(speed2);
                        return;
                    } else {
                        if (socketSpeedModel.getOnlyRead() == 1) {
                            this.mSocketProductModel.setVspeed(speed2);
                            return;
                        }
                        return;
                    }
                }
                return;
            }
            if ("setMode".equals(action)) {
                if (socketSpeedModel.isMotor1Mode()) {
                    if (socketSpeedModel.getOnlyRead() == 1) {
                        this.mSocketProductModel.setMode(socketSpeedModel.getMode());
                    }
                    if (socketSpeedModel.getMode() <= 0) {
                        if (socketSpeedModel.getOnlyRead() == 0 && this.mServerSwitch) {
                            BleManager.getInstance(this).writeSpeed(BleManager.getInstance(this).getmConnBleDevice().getSpeed());
                            return;
                        }
                        return;
                    }
                    if (ProductUtil.hasMode(0, socketSpeedModel.getMode())) {
                        if (socketSpeedModel.getOnlyRead() == 0 && this.mServerSwitch) {
                            BleManager.getInstance(this).writeData(ProductUtil.getProductMode(0)[socketSpeedModel.getMode() - 1].getmCommand());
                            return;
                        }
                        return;
                    }
                    if (socketSpeedModel.getOnlyRead() == 0 && this.mServerSwitch) {
                        BleManager.getInstance(this).writeSpeedMode(socketSpeedModel.getMode());
                        return;
                    }
                    return;
                }
                if (socketSpeedModel.isMotor2Mode()) {
                    if (socketSpeedModel.getOnlyRead() == 1) {
                        this.mSocketProductModel.setVibration(socketSpeedModel.getMode());
                    }
                    if (socketSpeedModel.getMode() == 0) {
                        if (socketSpeedModel.getOnlyRead() == 0 && this.mServerSwitch) {
                            if (BleManager.getInstance(this).getmConnBleDevice().getVibration() > 0) {
                                BleManager.getInstance(this).writeVibrate(BleManager.getInstance(this).getmConnBleDevice().getVibration());
                                return;
                            } else {
                                BleManager.getInstance(this).writeCloseVibrate();
                                return;
                            }
                        }
                        return;
                    }
                    if (ProductUtil.hasMode(1, socketSpeedModel.getMode())) {
                        if (socketSpeedModel.getOnlyRead() == 0 && this.mServerSwitch) {
                            BleManager.getInstance(this).writeData(ProductUtil.getProductMode(1)[socketSpeedModel.getMode() - 1].getmCommand());
                            return;
                        }
                        return;
                    }
                    if (socketSpeedModel.getOnlyRead() == 0 && this.mServerSwitch) {
                        BleManager.getInstance(this).writeVibrateMode(socketSpeedModel.getMode());
                        return;
                    }
                    return;
                }
                return;
            }
            if ("setClimax".equals(action)) {
                if (socketSpeedModel.getOnlyRead() == 0 && this.mServerSwitch) {
                    if ("1003".equals(MyApplication.getApplication().getDeviceCode())) {
                        BleManager.getInstance(this).writeOpenClimax();
                        return;
                    } else {
                        BleManager.getInstance(this).writeOpenClimax01();
                        return;
                    }
                }
                return;
            }
            if ("setStroke".equals(action)) {
                if (socketSpeedModel.getOnlyRead() == 1) {
                    this.mSocketProductModel.setStrokeMin(socketSpeedModel.getStrokeMin());
                    this.mSocketProductModel.setStrokeMax(socketSpeedModel.getStrokeMax());
                    return;
                } else {
                    if (socketSpeedModel.getOnlyRead() == 0 && this.mServerSwitch) {
                        new Thread(new Runnable() { // from class: com.hismith.hismithapp.service.HuDongPlayService.3
                            @Override // java.lang.Runnable
                            public void run() {
                                try {
                                    BleManager.getInstance(HuDongPlayService.this).writeStartPosition(socketSpeedModel.getStrokeMin());
                                    Thread.sleep(100L);
                                    BleManager.getInstance(HuDongPlayService.this).writeStopPosition(socketSpeedModel.getStrokeMax());
                                } catch (Exception e) {
                                    e.printStackTrace();
                                }
                            }
                        }).start();
                        return;
                    }
                    return;
                }
            }
            if ("setSmooth".equals(action)) {
                if (socketSpeedModel.getOnlyRead() == 0 && this.mServerSwitch) {
                    BleManager.getInstance(this).writeSmooth(socketSpeedModel.getSmooth());
                    return;
                } else {
                    if (socketSpeedModel.getOnlyRead() == 1) {
                        this.mSocketProductModel.setSmooth(socketSpeedModel.getSmooth());
                        return;
                    }
                    return;
                }
            }
            if ("setStrokeType".equals(action)) {
                if (socketSpeedModel.getOnlyRead() == 1) {
                    this.mSocketProductModel.setStrokeType(socketSpeedModel.getStrokeType());
                    return;
                }
                if (socketSpeedModel.getOnlyRead() == 0 && this.mServerSwitch) {
                    if (socketSpeedModel.getStrokeType() == 1) {
                        BleManager.getInstance(this).writeRunManual();
                        return;
                    } else {
                        BleManager.getInstance(this).writeRunAuto();
                        return;
                    }
                }
                return;
            }
            if ("setStrokeNow".equals(action)) {
                if (socketSpeedModel.getOnlyRead() == 1) {
                    this.mSocketProductModel.setStrokeNow(socketSpeedModel.getStrokeNow());
                    return;
                } else {
                    if (socketSpeedModel.getOnlyRead() == 0 && this.mServerSwitch) {
                        BleManager.getInstance(this).writePosition(socketSpeedModel.getStrokeNow());
                        return;
                    }
                    return;
                }
            }
            if ("setReversal".equals(action)) {
                if (socketSpeedModel.getOnlyRead() == 1) {
                    this.mSocketProductModel.setReversal(socketSpeedModel.getReversal());
                    return;
                }
                if (socketSpeedModel.getOnlyRead() == 0 && this.mServerSwitch) {
                    if (socketSpeedModel.getReversal() == 1) {
                        BleManager.getInstance(this).writeOpenReversal();
                        return;
                    } else {
                        if (socketSpeedModel.getReversal() == 0) {
                            BleManager.getInstance(this).writeCloseReversal();
                            return;
                        }
                        return;
                    }
                }
                return;
            }
            if ("runStroke".equals(action)) {
                Integer strokeStart = socketSpeedModel.getStrokeStart();
                final Integer strokeEnd = socketSpeedModel.getStrokeEnd();
                if (strokeStart != null && strokeStart.intValue() >= this.mStartPositionSafeMax) {
                    BleManager.getInstance(this).writeData(BleManager.getInstance(this).getBleCCData(7, strokeStart.intValue()));
                    i2 = 15;
                }
                if (strokeEnd == null || strokeEnd.intValue() > this.mStopPositionSafeMax) {
                    return;
                }
                this.mHandler.postDelayed(new Runnable() { // from class: com.hismith.hismithapp.service.HuDongPlayService.4
                    @Override // java.lang.Runnable
                    public void run() {
                        BleManager.getInstance(HuDongPlayService.this).writeData(BleManager.getInstance(HuDongPlayService.this).getBleCCData(8, strokeEnd.intValue()));
                    }
                }, i2);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void presenterSVGFile(SocketProductModel socketProductModel) {
        String str;
        LogUtil.e("SVG逻辑判断函数");
        if (socketProductModel == null || TextUtils.isEmpty(socketProductModel.getSvgName()) || TextUtils.isEmpty(socketProductModel.getSvgUrl())) {
            LogUtil.e("SVG信息数据中有空。无法下载");
            return;
        }
        String svgName = socketProductModel.getSvgName();
        if (socketProductModel.getSvgLeftRight() == 1) {
            svgName = socketProductModel.getSvgName() + "_left";
            str = socketProductModel.getSvgName() + "_right";
        } else {
            str = "";
        }
        long playSVGModifyTime = CacheUtil.getPlaySVGModifyTime(MyApplication.getApplication(), svgName);
        if (CacheUtil.playSVGExists(MyApplication.getApplication(), svgName) && ((socketProductModel.getSvgLeftRight() != 1 || CacheUtil.playSVGExists(MyApplication.getApplication(), str)) && (socketProductModel.getSvgVersion() <= 0 || socketProductModel.getSvgVersion() * 1000 <= playSVGModifyTime))) {
            LogUtil.e("SVG文件已存在，并且已经是最新版。无需下载");
            return;
        }
        if (CacheUtil.playSVGExists(MyApplication.getApplication(), svgName)) {
            LogUtil.e("SVG本地文件:" + playSVGModifyTime + ", server最后修改时间：" + (socketProductModel.getSvgVersion() * 1000));
            LogUtil.e("SVG文件时间判断不是最新版，需要重新下载");
        } else {
            LogUtil.e("SVG文件不存在，需要下载");
        }
        downloadSVGFile(svgName + ".svg", socketProductModel.getSvgUrl(), true);
        if (socketProductModel.getSvgLeftRight() == 1) {
            downloadSVGFile(str + ".svg", socketProductModel.getSvgUrl().replace("_left", "_right"), false);
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

    public static void startService(Context context, int i, int i2, String str) {
        Intent intent = new Intent(context, (Class<?>) HuDongPlayService.class);
        intent.putExtra("type", i);
        intent.putExtra("msgId", i2);
        intent.putExtra("name", str);
        context.startService(intent);
    }

    public static void startService(Context context, int i, int i2, String str, String str2) {
        Intent intent = new Intent(context, (Class<?>) HuDongPlayService.class);
        intent.putExtra("type", i);
        intent.putExtra("msgId", i2);
        intent.putExtra("name", str);
        intent.putExtra("gameToken", str2);
        context.startService(intent);
    }

    public static void startService(Context context, int i, int i2, String str, String str2, String str3) {
        Intent intent = new Intent(context, (Class<?>) HuDongPlayService.class);
        intent.putExtra("type", i);
        intent.putExtra("msgId", i2);
        intent.putExtra("name", str);
        intent.putExtra("headIcon", str2);
        intent.putExtra("gameToken", str3);
        context.startService(intent);
    }

    public static void stopService(Context context) {
        context.stopService(new Intent(context, (Class<?>) HuDongPlayService.class));
    }

    private void unregisterReceiver() {
        try {
            unregisterReceiver(this.mReceiver);
            LogUtil.e("unregisterReceiver()");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override // android.app.Service
    public IBinder onBind(Intent intent) {
        HuDongBinder huDongBinder = this.mBinder;
        if (huDongBinder != null) {
            return huDongBinder;
        }
        return null;
    }

    @Override // android.app.Service
    public void onCreate() {
        super.onCreate();
        this.mGson = new Gson();
        this.baseRequest = new RetrofitRequest();
        this.mBinder = new HuDongBinder();
        this.mSafeMax = ((Integer) SharedPreferencesUtil.get(this, Constans.SAFE_FUN1, 40)).intValue();
        this.mVibrateSafeMax = ((Integer) SharedPreferencesUtil.get(this, Constans.SAFE_FUN2, 40)).intValue();
    }

    @Override // android.app.Service
    public void onDestroy() {
        super.onDestroy();
        LogUtil.e("Service OnDestroy()");
        this.mBinder = null;
        mClientConnection = 0;
        mServerConnection = 0;
        unregisterReceiver();
    }

    @Override // android.app.Service
    public int onStartCommand(Intent intent, int i, int i2) {
        registerReceiver();
        if (intent != null) {
            this.type = intent.getIntExtra("type", 0);
            this.msgId = intent.getIntExtra("msgId", 0);
            this.name = intent.getStringExtra("name");
            this.headIcon = intent.getStringExtra("headIcon");
            this.gameToken = intent.getStringExtra("gameToken");
            int i3 = this.type;
            if (i3 == 0) {
                RemoteFloatingManager.getInstance().destroyFloatWindow();
                Intent intent2 = new Intent(this, (Class<?>) ControlSponsorActivity2.class);
                intent2.addFlags(268435456);
                startActivity(intent2);
                this.mHandler.sendEmptyMessageDelayed(1, this.TIME_NO_ANSWER);
            } else if (i3 == 1) {
                Intent intent3 = new Intent(this, (Class<?>) ControlAcceptActivity2.class);
                intent3.addFlags(268435456);
                startActivity(intent3);
                this.mHandler.sendEmptyMessageDelayed(1, this.TIME_NO_ANSWER);
            }
        }
        return 1;
    }

    @Override // android.app.Service
    public boolean onUnbind(Intent intent) {
        return super.onUnbind(intent);
    }
}
