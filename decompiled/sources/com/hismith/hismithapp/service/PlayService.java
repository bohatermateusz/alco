package com.hismith.hismithapp.service;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.os.Build;
import android.os.IBinder;
import android.os.SystemClock;
import android.widget.RemoteViews;
import androidx.core.app.NotificationCompat;
import androidx.media3.common.C;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.bluetooth.BleManager;
import com.hismith.hismithapp.product.ProductUtil;
import com.hismith.hismithapp.utils.AppUtil;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.ModeDataVCUtil;

/* loaded from: classes3.dex */
public class PlayService extends Service {
    private static final int NOTIFICATION_ID = 1;
    private Notification notification;
    private NotificationManager notificationManager;
    private RemoteViews views;
    private final String ACTION_PLAY = "com.hismith.hismithapp.action.ble.action.play";
    private final String NOTICE_ID_KEY = "com.hismith.hismithapp.action.ble.notice";
    private final int NOTICE_ID_START_RUN = 1;
    private final int NOTICE_ID_MODE_NEXT = 2;
    private final int NOTICE_ID_SPEED_UP = 3;
    private final int NOTICE_ID_SPEED_DOWN = 4;
    private BroadcastReceiver mReceiver = new BroadcastReceiver() { // from class: com.hismith.hismithapp.service.PlayService.1
        /* JADX WARN: Code restructure failed: missing block: B:53:0x0145, code lost:
        
            if (r0.equals("04") != false) goto L55;
         */
        /* JADX WARN: Code restructure failed: missing block: B:54:0x014e, code lost:
        
            r8.this$0.views.setTextViewText(com.hismith.hismithapp.R.id.txt_speed, "速度：" + java.lang.Integer.parseInt(r10, 16));
         */
        /* JADX WARN: Code restructure failed: missing block: B:56:0x014c, code lost:
        
            if (r0.equals("03") != false) goto L55;
         */
        /* JADX WARN: Failed to restore switch over string. Please report as a decompilation issue */
        /* JADX WARN: Failed to restore switch over string. Please report as a decompilation issue
        java.lang.NullPointerException: Cannot invoke "java.util.List.iterator()" because the return value of "jadx.core.dex.visitors.regions.SwitchOverStringVisitor$SwitchData.getNewCases()" is null
        	at jadx.core.dex.visitors.regions.SwitchOverStringVisitor.restoreSwitchOverString(SwitchOverStringVisitor.java:109)
        	at jadx.core.dex.visitors.regions.SwitchOverStringVisitor.visitRegion(SwitchOverStringVisitor.java:66)
        	at jadx.core.dex.visitors.regions.DepthRegionTraversal.traverseIterativeStepInternal(DepthRegionTraversal.java:77)
        	at jadx.core.dex.visitors.regions.DepthRegionTraversal.traverseIterativeStepInternal(DepthRegionTraversal.java:82)
         */
        @Override // android.content.BroadcastReceiver
        /*
            Code decompiled incorrectly, please refer to instructions dump.
        */
        public void onReceive(Context context, Intent intent) {
            int parseInt;
            int i;
            int i2;
            String action = intent.getAction();
            if ("com.hismith.hismithapp.action.ble.action.play".equals(action)) {
                int intExtra = intent.getIntExtra("com.hismith.hismithapp.action.ble.notice", 0);
                if (intExtra == 1) {
                    BleManager.getInstance(PlayService.this).writeShutdown();
                    return;
                }
                if (intExtra == 2) {
                    try {
                        int mode = BleManager.getInstance(PlayService.this).getmConnBleDevice().getMode();
                        if (ProductUtil.hasMode(0, mode)) {
                            BleManager.getInstance(PlayService.this).writeSpeedMode(mode);
                            return;
                        }
                        return;
                    } catch (Exception e) {
                        e.printStackTrace();
                        return;
                    }
                }
                if (intExtra == 3) {
                    int speed = BleManager.getInstance(PlayService.this).getmConnBleDevice().getSpeed() + 5;
                    i = speed >= 0 ? speed : 0;
                    i2 = i <= 100 ? i : 100;
                    LogUtil.e("NOTICE_ID_SPEED_UP:" + i2);
                    BleManager.getInstance(PlayService.this).writeSpeed(i2);
                    return;
                }
                if (intExtra != 4) {
                    return;
                }
                int speed2 = BleManager.getInstance(PlayService.this).getmConnBleDevice().getSpeed() - 5;
                i = speed2 >= 0 ? speed2 : 0;
                i2 = i <= 100 ? i : 100;
                LogUtil.e("NOTICE_ID_SPEED_DOWN:" + i2);
                BleManager.getInstance(PlayService.this).writeSpeed(i2);
                return;
            }
            if (Constans.RECEIVER_BLE_RECE_DATA.equals(action)) {
                String stringExtra = intent.getStringExtra(Constans.BleType);
                String stringExtra2 = intent.getStringExtra(Constans.BleData0);
                String stringExtra3 = intent.getStringExtra(Constans.BleData1);
                LogUtil.e("广播数据" + stringExtra + stringExtra2 + stringExtra3);
                if ("AA".equals(stringExtra)) {
                    switch (stringExtra2.hashCode()) {
                        case 1538:
                            if (stringExtra2.equals("02")) {
                                PlayService.this.views.setTextViewText(R.id.txt_speed, "速度：0");
                                break;
                            }
                            break;
                        case 1541:
                            if (stringExtra2.equals("05") && (parseInt = Integer.parseInt(stringExtra3, 16)) > 0) {
                                PlayService.this.views.setTextViewText(R.id.txt_speed, "模式：" + parseInt);
                                break;
                            }
                            break;
                        case 1542:
                            stringExtra2.equals("06");
                            break;
                        case 1543:
                            stringExtra2.equals("07");
                            break;
                    }
                    PlayService.this.notificationManager.notify(1, PlayService.this.notification);
                    return;
                }
                if ("CA".equals(stringExtra) || "CC".equals(stringExtra)) {
                    switch (stringExtra2.hashCode()) {
                        case 1537:
                            stringExtra2.equals(ModeDataVCUtil.HEAD_VERSION);
                            break;
                        case 1538:
                            stringExtra2.equals("02");
                            break;
                        case 1539:
                            stringExtra2.equals("03");
                            break;
                        case 1541:
                            stringExtra2.equals("05");
                            break;
                        case 1542:
                            stringExtra2.equals("06");
                            break;
                        case 1543:
                            stringExtra2.equals("07");
                            break;
                    }
                }
            }
        }
    };

    private Notification createForegroundNotification() {
        this.notificationManager = (NotificationManager) getSystemService("notification");
        NotificationChannel notificationChannel = new NotificationChannel("notification_channel_id_01", "Foreground Notification", 4);
        notificationChannel.setDescription("Channel description");
        notificationChannel.enableLights(true);
        notificationChannel.setVibrationPattern(new long[]{0, 1000, 500, 1000});
        notificationChannel.enableVibration(true);
        NotificationManager notificationManager = this.notificationManager;
        if (notificationManager != null) {
            notificationManager.createNotificationChannel(notificationChannel);
        }
        NotificationCompat.Builder builder = new NotificationCompat.Builder(this, "notification_channel_id_01");
        builder.setSmallIcon(R.mipmap.ic_128_128);
        builder.setCustomContentView(this.views);
        builder.setWhen(System.currentTimeMillis());
        Intent intent = new Intent();
        builder.setContentIntent(Build.VERSION.SDK_INT >= 31 ? PendingIntent.getActivity(this, 1, intent, 67108864) : PendingIntent.getActivity(this, 1, intent, C.BUFFER_FLAG_FIRST_SAMPLE));
        return builder.build();
    }

    private void notificationNextModeClick() {
        int uptimeMillis = (int) SystemClock.uptimeMillis();
        Intent intent = new Intent("com.hismith.hismithapp.action.ble.action.play");
        intent.putExtra("com.hismith.hismithapp.action.ble.notice", 2);
        this.views.setOnClickPendingIntent(R.id.layout_next, Build.VERSION.SDK_INT >= 31 ? PendingIntent.getBroadcast(this, uptimeMillis, intent, 67108864) : PendingIntent.getBroadcast(this, uptimeMillis, intent, C.BUFFER_FLAG_FIRST_SAMPLE));
    }

    private void notificationSetPlayClick() {
        int uptimeMillis = (int) SystemClock.uptimeMillis();
        Intent intent = new Intent("com.hismith.hismithapp.action.ble.action.play");
        intent.putExtra("com.hismith.hismithapp.action.ble.notice", 1);
        this.views.setOnClickPendingIntent(R.id.layout_play, Build.VERSION.SDK_INT >= 31 ? PendingIntent.getBroadcast(this, uptimeMillis, intent, 67108864) : PendingIntent.getBroadcast(this, uptimeMillis, intent, C.BUFFER_FLAG_FIRST_SAMPLE));
    }

    private void notificationSpeedDownClick() {
        int uptimeMillis = (int) SystemClock.uptimeMillis();
        Intent intent = new Intent("com.hismith.hismithapp.action.ble.action.play");
        intent.putExtra("com.hismith.hismithapp.action.ble.notice", 4);
        this.views.setOnClickPendingIntent(R.id.layout_down, Build.VERSION.SDK_INT >= 31 ? PendingIntent.getBroadcast(this, uptimeMillis, intent, 67108864) : PendingIntent.getBroadcast(this, uptimeMillis, intent, C.BUFFER_FLAG_FIRST_SAMPLE));
    }

    private void notificationSpeedUpClick() {
        int uptimeMillis = (int) SystemClock.uptimeMillis();
        Intent intent = new Intent("com.hismith.hismithapp.action.ble.action.play");
        intent.putExtra("com.hismith.hismithapp.action.ble.notice", 3);
        this.views.setOnClickPendingIntent(R.id.layout_up, Build.VERSION.SDK_INT >= 31 ? PendingIntent.getBroadcast(this, uptimeMillis, intent, 67108864) : PendingIntent.getBroadcast(this, uptimeMillis, intent, C.BUFFER_FLAG_FIRST_SAMPLE));
    }

    private void registerReceiver() {
        IntentFilter intentFilter = new IntentFilter();
        intentFilter.addAction(Constans.RECEIVER_BLE_RECE_DATA);
        intentFilter.addAction("com.hismith.hismithapp.action.ble.action.play");
        AppUtil.registerReceiver(this, this.mReceiver, intentFilter);
        LogUtil.e("绑定广播");
    }

    private void unregisterReceiver() {
        try {
            unregisterReceiver(this.mReceiver);
            LogUtil.e("解除广播");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override // android.app.Service
    public IBinder onBind(Intent intent) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override // android.app.Service
    public void onCreate() {
        super.onCreate();
        registerReceiver();
        this.views = new RemoteViews(getPackageName(), R.layout.notification_play);
        notificationSetPlayClick();
        notificationNextModeClick();
        notificationSpeedUpClick();
        notificationSpeedDownClick();
        Notification createForegroundNotification = createForegroundNotification();
        this.notification = createForegroundNotification;
        startForeground(1, createForegroundNotification);
        LogUtil.e("开启前台服务");
    }

    @Override // android.app.Service
    public void onDestroy() {
        super.onDestroy();
        LogUtil.e("关闭前台服务");
        stopForeground(true);
        unregisterReceiver();
    }

    @Override // android.app.Service
    public int onStartCommand(Intent intent, int i, int i2) {
        return 1;
    }

    @Override // android.app.Service
    public boolean onUnbind(Intent intent) {
        return super.onUnbind(intent);
    }
}
