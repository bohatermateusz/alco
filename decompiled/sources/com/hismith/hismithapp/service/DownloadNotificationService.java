package com.hismith.hismithapp.service;

import android.app.IntentService;
import android.app.NotificationManager;
import android.content.Intent;
import androidx.core.app.NotificationCompat;

/* loaded from: classes.dex */
public class DownloadNotificationService extends IntentService {
    private NotificationCompat.Builder notificationBuilder;
    private NotificationManager notificationManager;

    public DownloadNotificationService(String str) {
        super(str);
    }

    @Override // android.app.IntentService
    protected void onHandleIntent(Intent intent) {
    }
}
