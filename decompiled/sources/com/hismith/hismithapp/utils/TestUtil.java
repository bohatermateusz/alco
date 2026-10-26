package com.hismith.hismithapp.utils;

import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.content.Context;
import android.content.Intent;
import android.os.Build;
import android.text.TextUtils;
import androidx.core.app.NotificationCompat;
import androidx.media3.common.C;
import com.hismith.hismithapp.MainActivity;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.application.MyApplication;

/* loaded from: classes2.dex */
public class TestUtil {
    public static void sendNotification(Context context, String str, String str2) {
        try {
            if (!TextUtils.isEmpty(str2)) {
                str2 = str2.replace("[", "").replace("]", "");
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        NotificationManager notificationManager = (NotificationManager) context.getSystemService("notification");
        Intent intent = new Intent(context, (Class<?>) MainActivity.class);
        PendingIntent broadcast = Build.VERSION.SDK_INT >= 31 ? PendingIntent.getBroadcast(MyApplication.getApplication(), 0, intent, 67108864) : PendingIntent.getBroadcast(MyApplication.getApplication(), 0, intent, C.BUFFER_FLAG_FIRST_SAMPLE);
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
}
