package com.hismith.hismithapp.utils;

import android.content.Context;
import android.content.Intent;
import com.hismith.hismithapp.PersonalDetailsActivity2;
import com.hismith.hismithapp.UserDetailsActivity2;
import com.hismith.hismithapp.application.MyApplication;

/* loaded from: classes3.dex */
public class IntentUtil {
    public static void startUserInfoActivity(Context context, int i) {
        if (MyApplication.getApplication().getmUserModel() != null && MyApplication.getApplication().getmUserModel().getUserId() == i) {
            context.startActivity(new Intent(context, (Class<?>) PersonalDetailsActivity2.class));
            return;
        }
        Intent intent = new Intent(context, (Class<?>) UserDetailsActivity2.class);
        intent.putExtra("userId", i);
        context.startActivity(intent);
    }
}
