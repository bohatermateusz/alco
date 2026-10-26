package com.hismith.hismithapp;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.RelativeLayout;
import android.widget.TextView;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.recyclerview.widget.RecyclerView;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.utils.AppUtil;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.view.CircleImageView;
import com.hismith.hismithapp.view.CustomSeekbar;
import com.hismith.hismithapp.view.CustomWebView;
import com.hismith.hismithapp.view.PlayView_Four;
import com.hismith.hismithapp.view.VerticalProgressBar;

/* loaded from: classes3.dex */
public class LineRoomActivity2 extends BaseActivity {
    private Button btn_message_switch;
    private ImageView img_position;
    private ImageView img_run_mode;
    private FrameLayout layout_position_max;
    private FrameLayout layout_position_min;
    private ConstraintLayout layout_sms_touch_position;
    private FrameLayout layout_speed_max;
    private ConstraintLayout layout_touch_position;
    private EditText line_room_edit_message;
    private FrameLayout line_room_fragment_preview;
    private ImageView line_room_img_chat;
    private ImageView line_room_img_exit;
    private ImageView line_room_img_mode;
    private CircleImageView line_room_img_owner_head;
    private ImageView line_room_img_pause;
    private ImageView line_room_img_room_public;
    private ImageView line_room_img_sound;
    private ImageView line_room_img_start;
    private ImageView line_room_img_user_crown;
    private ImageView line_room_img_user_invite;
    private ImageView line_room_img_voice;
    private ImageView line_room_layout_bg;
    private FrameLayout line_room_layout_max_speed;
    private FrameLayout line_room_layout_max_vibrate;
    private LinearLayout line_room_layout_message;
    private LinearLayout line_room_layout_sms_info;
    private RelativeLayout line_room_layout_speed_info;
    private LinearLayout line_room_layout_user_count;
    private RelativeLayout line_room_layout_user_info;
    private RelativeLayout line_room_layout_vibrate_info;
    private View line_room_line;
    private PlayView_Four line_room_playView;
    private PlayView_Four line_room_playView_vibrate;
    private VerticalProgressBar line_room_progress_speed;
    private VerticalProgressBar line_room_progress_vibrate;
    private RecyclerView line_room_recycler_message;
    private RecyclerView line_room_recycler_user;
    private TextView line_room_txt_max_speed;
    private TextView line_room_txt_max_vibrate;
    private TextView line_room_txt_owner_name;
    private TextView line_room_txt_room_id;
    private TextView line_room_txt_speed;
    private TextView line_room_txt_user_count;
    private TextView line_room_txt_vibrate;
    private BroadcastReceiver mReceiver = new BroadcastReceiver() { // from class: com.hismith.hismithapp.LineRoomActivity2.1
        @Override // android.content.BroadcastReceiver
        public void onReceive(Context context, Intent intent) {
            if (Constans.RECEIVER_BLE_RECE_DATA.equals(intent.getAction())) {
                String stringExtra = intent.getStringExtra(Constans.BleType);
                String stringExtra2 = intent.getStringExtra(Constans.BleData0);
                String stringExtra3 = intent.getStringExtra(Constans.BleData1);
                LogUtil.e("蓝牙广播数据接收：" + stringExtra + ", data0:" + stringExtra2 + ", data1:" + stringExtra3);
                if ("AA".equals(stringExtra)) {
                    LineRoomActivity2.this.initBleAAData(stringExtra2, stringExtra3);
                } else if ("CA".equals(stringExtra) || "CC".equals(stringExtra)) {
                    LineRoomActivity2.this.initBleCCData(stringExtra, stringExtra2, stringExtra3);
                }
            }
        }
    };
    private ProgressBar progress_position;
    private ProgressBar progress_speed;
    private CustomSeekbar seek_smooth;
    private TextView txt_position;
    private TextView txt_position_label;
    private TextView txt_position_max;
    private TextView txt_position_min;
    private TextView txt_smooth;
    private TextView txt_smooth_label;
    private TextView txt_speed;
    private TextView txt_speed_label;
    private TextView txt_speed_max;
    private View view_network_error;
    private CustomWebView web_view;
    private CustomWebView web_view_rocket;

    private void init() {
        registerReceiver();
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void initBleAAData(String str, String str2) {
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void initBleCCData(String str, String str2, String str3) {
    }

    private void initView() {
        this.view_network_error = findViewById(R.id.view_network_error);
        this.line_room_layout_bg = (ImageView) findViewById(R.id.line_room_layout_bg);
        this.line_room_playView = (PlayView_Four) findViewById(R.id.line_room_playView);
        this.line_room_playView_vibrate = (PlayView_Four) findViewById(R.id.line_room_playView_vibrate);
        this.line_room_line = findViewById(R.id.line_room_line);
        this.line_room_layout_user_info = (RelativeLayout) findViewById(R.id.line_room_layout_user_info);
        this.line_room_img_owner_head = (CircleImageView) findViewById(R.id.line_room_img_owner_head);
        this.line_room_txt_owner_name = (TextView) findViewById(R.id.line_room_txt_owner_name);
        this.line_room_txt_room_id = (TextView) findViewById(R.id.line_room_txt_room_id);
        this.line_room_img_room_public = (ImageView) findViewById(R.id.line_room_img_room_public);
        this.line_room_recycler_user = (RecyclerView) findViewById(R.id.line_room_recycler_user);
        this.line_room_layout_user_count = (LinearLayout) findViewById(R.id.line_room_layout_user_count);
        this.line_room_txt_user_count = (TextView) findViewById(R.id.line_room_txt_user_count);
        this.line_room_img_user_invite = (ImageView) findViewById(R.id.line_room_img_user_invite);
        this.line_room_img_user_crown = (ImageView) findViewById(R.id.line_room_img_user_crown);
        this.line_room_img_mode = (ImageView) findViewById(R.id.line_room_img_mode);
        this.line_room_fragment_preview = (FrameLayout) findViewById(R.id.line_room_fragment_preview);
        this.line_room_layout_speed_info = (RelativeLayout) findViewById(R.id.line_room_layout_speed_info);
        this.line_room_progress_speed = (VerticalProgressBar) findViewById(R.id.line_room_progress_speed);
        this.line_room_layout_max_speed = (FrameLayout) findViewById(R.id.line_room_layout_max_speed);
        this.line_room_txt_max_speed = (TextView) findViewById(R.id.line_room_txt_max_speed);
        this.line_room_txt_speed = (TextView) findViewById(R.id.line_room_txt_speed);
        this.line_room_layout_vibrate_info = (RelativeLayout) findViewById(R.id.line_room_layout_vibrate_info);
        this.line_room_progress_vibrate = (VerticalProgressBar) findViewById(R.id.line_room_progress_vibrate);
        this.line_room_layout_max_vibrate = (FrameLayout) findViewById(R.id.line_room_layout_max_vibrate);
        this.line_room_txt_max_vibrate = (TextView) findViewById(R.id.line_room_txt_max_vibrate);
        this.line_room_txt_vibrate = (TextView) findViewById(R.id.line_room_txt_vibrate);
        this.line_room_layout_sms_info = (LinearLayout) findViewById(R.id.line_room_layout_sms_info);
        this.img_run_mode = (ImageView) findViewById(R.id.img_run_mode);
        this.txt_speed_label = (TextView) findViewById(R.id.txt_speed_label);
        this.txt_speed = (TextView) findViewById(R.id.txt_speed);
        this.txt_speed_max = (TextView) findViewById(R.id.txt_speed_max);
        this.txt_position_label = (TextView) findViewById(R.id.txt_position_label);
        this.txt_position = (TextView) findViewById(R.id.txt_position);
        this.txt_position_max = (TextView) findViewById(R.id.txt_position_max);
        this.txt_position_min = (TextView) findViewById(R.id.txt_position_min);
        this.txt_smooth_label = (TextView) findViewById(R.id.txt_smooth_label);
        this.txt_smooth = (TextView) findViewById(R.id.txt_smooth);
        this.progress_speed = (ProgressBar) findViewById(R.id.progress_speed);
        this.progress_position = (ProgressBar) findViewById(R.id.progress_position);
        this.seek_smooth = (CustomSeekbar) findViewById(R.id.seek_smooth);
        this.layout_speed_max = (FrameLayout) findViewById(R.id.layout_speed_max);
        this.layout_position_max = (FrameLayout) findViewById(R.id.layout_position_max);
        this.layout_position_min = (FrameLayout) findViewById(R.id.layout_position_min);
        this.layout_sms_touch_position = (ConstraintLayout) findViewById(R.id.layout_sms_touch_position);
        this.layout_touch_position = (ConstraintLayout) findViewById(R.id.layout_touch_position);
        this.img_position = (ImageView) findViewById(R.id.img_position);
        this.line_room_layout_message = (LinearLayout) findViewById(R.id.line_room_layout_message);
        this.line_room_img_sound = (ImageView) findViewById(R.id.line_room_img_sound);
        this.line_room_img_voice = (ImageView) findViewById(R.id.line_room_img_voice);
        this.line_room_img_chat = (ImageView) findViewById(R.id.line_room_img_chat);
        this.line_room_img_exit = (ImageView) findViewById(R.id.line_room_img_exit);
        this.line_room_edit_message = (EditText) findViewById(R.id.line_room_edit_message);
        this.line_room_recycler_message = (RecyclerView) findViewById(R.id.line_room_recycler_message);
        this.btn_message_switch = (Button) findViewById(R.id.btn_message_switch);
        this.line_room_img_start = (ImageView) findViewById(R.id.line_room_img_start);
        this.web_view_rocket = (CustomWebView) findViewById(R.id.web_view_rocket);
        this.web_view = (CustomWebView) findViewById(R.id.web_view);
    }

    private void registerReceiver() {
        IntentFilter intentFilter = new IntentFilter();
        intentFilter.addAction(Constans.RECEIVER_BLE_RECE_DATA);
        AppUtil.registerReceiver(this, this.mReceiver, intentFilter);
    }

    private void setListener() {
    }

    private void setPositionText(int i) {
        this.progress_position.setProgress(i);
        this.txt_position.setText(i + "");
    }

    private void setSpeedText(int i) {
        if (MyApplication.getApplication().getmProductModel() == null || !Constans.DEVICE_TYPE_SMS.equals(MyApplication.getApplication().getmProductModel().getDeviceType())) {
            this.line_room_progress_speed.setProgress(i);
            this.line_room_txt_speed.setText(i + "");
        } else {
            this.progress_speed.setProgress(i);
            this.txt_speed.setText(i + "");
        }
    }

    private void setSpeedTextService(int i) {
        if (MyApplication.getApplication().getmProductModel() == null || !Constans.DEVICE_TYPE_SMS.equals(MyApplication.getApplication().getmProductModel().getDeviceType())) {
            this.line_room_progress_speed.setSecondaryProgress(i);
        } else {
            this.progress_speed.setSecondaryProgress(i);
        }
    }

    private void setVibrateText(int i) {
        this.line_room_progress_vibrate.setProgress(i);
        this.line_room_txt_vibrate.setText(i + "");
    }

    private void setVibrateTextService(int i) {
        this.line_room_progress_vibrate.setSecondaryProgress(i);
    }

    private void unRegisterReceiver() {
        try {
            unregisterReceiver(this.mReceiver);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, androidx.core.app.ComponentActivity, android.app.Activity
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        setContentView(R.layout.activity_line_room);
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.appcompat.app.AppCompatActivity, androidx.fragment.app.FragmentActivity, android.app.Activity
    protected void onDestroy() {
        super.onDestroy();
        unRegisterReceiver();
    }
}
