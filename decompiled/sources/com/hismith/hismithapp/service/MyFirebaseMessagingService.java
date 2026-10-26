package com.hismith.hismithapp.service;

import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.content.Context;
import android.content.Intent;
import android.os.Build;
import android.os.Bundle;
import android.text.TextUtils;
import androidx.core.app.NotificationCompat;
import androidx.media3.common.C;
import androidx.media3.extractor.text.ttml.TtmlNode;
import com.google.firebase.messaging.FirebaseMessagingService;
import com.google.firebase.messaging.RemoteMessage;
import com.hismith.hismithapp.MainActivity;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.application.MyLifecycleHandler;
import com.hismith.hismithapp.utils.LogUtil;
import java.util.Map;

/* loaded from: classes2.dex */
public class MyFirebaseMessagingService extends FirebaseMessagingService {
    private final String TAG = "MyFirebaseMessagingService";

    private void sendNotification(Context context, String str, String str2, Map<String, String> map) {
        try {
            if (!TextUtils.isEmpty(str2)) {
                str2 = str2.replace("[", "").replace("]", "");
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        NotificationManager notificationManager = (NotificationManager) context.getSystemService("notification");
        Intent intent = new Intent(this, (Class<?>) MainActivity.class);
        if (map != null && !map.isEmpty()) {
            Bundle bundle = new Bundle();
            for (String str3 : map.keySet()) {
                LogUtil.e("onMessageReceived bundle key:" + str3 + ", value:" + map.get(str3));
                bundle.putString(str3, map.get(str3));
            }
            intent.putExtras(bundle);
            LogUtil.e("onMessageReceived intent:" + intent.toString());
        }
        PendingIntent broadcast = Build.VERSION.SDK_INT >= 31 ? PendingIntent.getBroadcast(getApplicationContext(), 0, intent, 67108864) : PendingIntent.getBroadcast(getApplicationContext(), 0, intent, C.BUFFER_FLAG_FIRST_SAMPLE);
        notificationManager.createNotificationChannel(new NotificationChannel("static", "Primary Channel", 4));
        NotificationCompat.Builder builder = new NotificationCompat.Builder(context, "static");
        builder.setSmallIcon(R.mipmap.ic_128_128);
        builder.setContentTitle(str);
        builder.setAutoCancel(true);
        builder.setContentText(str2);
        builder.setWhen(System.currentTimeMillis());
        builder.setVibrate(new long[]{500, 500, 500, 500, 500});
        builder.setLights(-16776961, 1, 1);
        builder.setDefaults(3);
        builder.setContentIntent(broadcast);
        if (notificationManager != null) {
            notificationManager.notify(123, builder.build());
        }
    }

    @Override // com.google.firebase.messaging.FirebaseMessagingService
    public void onDeletedMessages() {
        super.onDeletedMessages();
    }

    @Override // com.google.firebase.messaging.FirebaseMessagingService
    public void onMessageReceived(RemoteMessage remoteMessage) {
        super.onMessageReceived(remoteMessage);
        if (MyLifecycleHandler.isApplicationInForeground()) {
            return;
        }
        if (remoteMessage.getNotification() == null || remoteMessage.getNotification().getBody() == null) {
            sendNotification(getApplicationContext(), remoteMessage.getData().get("title"), remoteMessage.getData().get(TtmlNode.TAG_BODY), remoteMessage.getData());
        } else {
            sendNotification(getApplicationContext(), remoteMessage.getNotification().getTitle(), remoteMessage.getNotification().getBody(), remoteMessage.getData());
        }
    }

    @Override // com.google.firebase.messaging.FirebaseMessagingService
    public void onMessageSent(String str) {
        super.onMessageSent(str);
    }

    @Override // com.google.firebase.messaging.FirebaseMessagingService
    public void onNewToken(String str) {
        super.onNewToken(str);
        MyApplication.FCM_TOKEN = str;
    }

    @Override // com.google.firebase.messaging.FirebaseMessagingService
    public void onSendError(String str, Exception exc) {
        super.onSendError(str, exc);
    }
}
