package com.hismith.hismithapp;

import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.text.TextUtils;
import android.widget.ImageView;
import com.hismith.hismithapp.utils.CacheUtil;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import pl.droidsonroids.gif.GifImageView;

/* loaded from: classes3.dex */
public class LunchActivity extends BaseActivity {
    private GifImageView gifImageView;
    private ImageView imgLunch;
    private Handler mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.LunchActivity.1
        @Override // android.os.Handler.Callback
        public boolean handleMessage(Message message) {
            int i = message.what;
            if (i == 0) {
                LunchActivity.this.startActivity(new Intent(LunchActivity.this, (Class<?>) LoginActivity.class));
                LunchActivity.this.finish();
                return false;
            }
            if (i != 1) {
                if (i != 2) {
                    return false;
                }
                LunchActivity.this.startActivity(new Intent(LunchActivity.this, (Class<?>) AdsActivity.class));
                LunchActivity.this.finish();
                return false;
            }
            Intent intent = new Intent(LunchActivity.this, (Class<?>) MainActivity.class);
            if (LunchActivity.this.getIntent() != null && LunchActivity.this.getIntent().getExtras() != null) {
                intent.putExtras(LunchActivity.this.getIntent().getExtras());
            }
            LunchActivity.this.startActivity(intent);
            LunchActivity.this.finish();
            return false;
        }
    });
    private int animTime = 2000;

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, androidx.core.app.ComponentActivity, android.app.Activity
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        setContentView(R.layout.activity_lunch);
        if (CacheUtil.adsExists(this)) {
            this.mHandler.sendEmptyMessageDelayed(2, this.animTime);
            return;
        }
        boolean isEmpty = TextUtils.isEmpty((String) SharedPreferencesUtil.get(this, Constans.Token, ""));
        Handler handler = this.mHandler;
        if (isEmpty) {
            handler.sendEmptyMessageDelayed(0, this.animTime);
        } else {
            handler.sendEmptyMessageDelayed(1, this.animTime);
        }
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.appcompat.app.AppCompatActivity, androidx.fragment.app.FragmentActivity, android.app.Activity
    protected void onDestroy() {
        super.onDestroy();
        Handler handler = this.mHandler;
        if (handler != null) {
            handler.removeCallbacksAndMessages(null);
        }
    }
}
