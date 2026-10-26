package com.hismith.hismithapp.utils;

import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.content.Context;
import android.content.ContextWrapper;
import android.content.Intent;
import android.os.Build;
import androidx.core.app.NotificationCompat;
import androidx.core.app.NotificationManagerCompat;
import androidx.media3.common.C;

/* loaded from: classes.dex */
public class NotificationUtil extends ContextWrapper {
    private NotificationCompat.Builder builder;
    private String channelId;
    private String content;
    private NotificationManager notificationManager;
    private NotificationManagerCompat notificationManagerCompat;
    private int notifyId;
    private int smallIcon;
    private String title;

    public NotificationUtil(Context context, int i, int i2, String str, String str2) {
        this(context, i, null, i2, str, str2);
    }

    public NotificationUtil(Context context, int i, String str, int i2, String str2, String str3) {
        super(context);
        this.channelId = "1";
        this.notifyId = i;
        this.channelId = str == null ? this.notifyId + "" : str;
        this.smallIcon = i2;
        this.title = str2;
        this.content = str3;
        baseNotification();
    }

    public NotificationUtil(Context context, int i, String str, String str2) {
        this(context, 1, null, i, str, str2);
    }

    private void baseNotification() {
        NotificationCompat.Builder builder = getBuilder(getApplicationContext(), this.channelId);
        this.builder = builder;
        builder.setSmallIcon(this.smallIcon);
        this.builder.setContentTitle(this.title);
        this.builder.setContentText(this.content);
    }

    private NotificationCompat.Builder getBuilder(Context context) {
        NotificationCompat.Builder builder = new NotificationCompat.Builder(context);
        this.builder = builder;
        return builder;
    }

    private NotificationCompat.Builder getBuilder(Context context, String str) {
        NotificationCompat.Builder builder = new NotificationCompat.Builder(context, str);
        this.builder = builder;
        return builder;
    }

    private void getNotificationManager() {
        this.notificationManager = (NotificationManager) getSystemService("notification");
        this.notificationManager.createNotificationChannel(new NotificationChannel(this.channelId, "channel_name", 3));
    }

    private void getNotificationManagerCompat() {
        this.notificationManagerCompat = NotificationManagerCompat.from(getApplicationContext());
    }

    private void notify(NotificationCompat.Builder builder) {
        getNotificationManager();
        this.notificationManager.notify(this.notifyId, builder.build());
    }

    public void cancel(int i) {
        this.notificationManager.cancel(i);
    }

    public NotificationCompat.Builder getBuilder() {
        return this.builder;
    }

    public NotificationManager getManager() {
        return this.notificationManager;
    }

    public void notify(Intent intent, String str) {
        this.builder.setContentIntent(Build.VERSION.SDK_INT >= 31 ? PendingIntent.getActivity(getApplicationContext(), 0, intent, 67108864) : PendingIntent.getActivity(getApplicationContext(), 0, intent, C.BUFFER_FLAG_FIRST_SAMPLE));
        this.builder.setContentText(str);
        notifyed();
    }

    public void notifyProgress(int i, int i2, String str) {
        NotificationCompat.Builder builder = this.builder;
        if (builder == null || i2 <= -1) {
            return;
        }
        builder.setContentText(str);
        this.builder.setProgress(i, i2, false);
        notifyed();
    }

    public void notifyed() {
        notify(this.builder);
    }
}
