package com.hismith.hismithapp;

import android.content.BroadcastReceiver;
import android.content.ContentValues;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.text.TextUtils;
import android.view.View;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.exifinterface.media.ExifInterface;
import androidx.fragment.app.FragmentActivity;
import androidx.fragment.app.FragmentTransaction;
import com.bumptech.glide.Glide;
import com.google.firebase.analytics.FirebaseAnalytics;
import com.google.gson.Gson;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.bluetooth.BleManager;
import com.hismith.hismithapp.db.DBMessageUtil;
import com.hismith.hismithapp.db.DatabaseHelper;
import com.hismith.hismithapp.fragment.PlaySMFragment;
import com.hismith.hismithapp.fragment.PlaySMSFragment;
import com.hismith.hismithapp.fragment.PlaySMSUIFragment;
import com.hismith.hismithapp.fragment.PlaySMUIFragment;
import com.hismith.hismithapp.fragment.SocketPlayFragment;
import com.hismith.hismithapp.http.HttpModel;
import com.hismith.hismithapp.http.RetrofitRequest;
import com.hismith.hismithapp.http.RxSubscribe;
import com.hismith.hismithapp.http.SocketManager;
import com.hismith.hismithapp.model.MessageModel;
import com.hismith.hismithapp.model.MessageModel.Message;
import com.hismith.hismithapp.model.MsgSendModel;
import com.hismith.hismithapp.model.SocketProductModel;
import com.hismith.hismithapp.model.SocketSpeedModel;
import com.hismith.hismithapp.model.UserModel;
import com.hismith.hismithapp.utils.AppUtil;
import com.hismith.hismithapp.utils.CheckUtil;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.DialogUtil;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import com.hismith.hismithapp.utils.ToastUtil;
import com.hismith.hismithapp.view.CircleImageView;
import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.schedulers.Schedulers;
import java.util.Objects;
import okhttp3.Response;

/* loaded from: classes3.dex */
public class ControlSponsorActivity extends BaseActivity implements View.OnClickListener, SocketManager.OnSocketNotifyListener, BleManager.OnConnectStateListener {
    private static String EXTRA_USER_ID = "extra_user_id";
    private static final int WHAT_NO_ANSWER = 1;
    private RetrofitRequest baseRequest;
    private DatabaseHelper dbHepler;
    private SQLiteDatabase dbRead;
    private SQLiteDatabase dbWrite;
    private CircleImageView img_logo;
    private ImageView img_refuse;
    private ImageView img_small;
    private ImageView iplay_layout_bg;
    private ConstraintLayout layout;
    private LinearLayout layout_control;
    private FrameLayout layout_frame;
    private LinearLayout layout_refuse;
    private Gson mGson;
    private MessageModel.Message mMessage;
    private SocketProductModel mSocketProductModel;
    private int mUserId;
    private UserModel mUserModel;
    private TextView title_txt_name;
    private TextView title_txt_status;
    private TextView txt_log;
    private TextView txt_refuse;
    private SocketPlayFragment mFragment = null;
    private int mSpeed = 0;
    private int mSafeMax = 100;
    private String new_device_type = Constans.DEVICE_TYPE_SMS;
    private final int REQUEST_SMALL = 101;
    private Handler mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.ControlSponsorActivity.1
        @Override // android.os.Handler.Callback
        public boolean handleMessage(Message message) {
            if (message.what != 1) {
                return false;
            }
            ControlSponsorActivity controlSponsorActivity = ControlSponsorActivity.this;
            ToastUtil.showShortToast(controlSponsorActivity, LanguageUtil.getLanguageByKey(controlSponsorActivity, "otherNotReceive"));
            ControlSponsorActivity.this.closeSocket();
            ControlSponsorActivity.this.finish();
            return false;
        }
    });
    private BroadcastReceiver mReceiver = new BroadcastReceiver() { // from class: com.hismith.hismithapp.ControlSponsorActivity.6
        @Override // android.content.BroadcastReceiver
        public void onReceive(Context context, Intent intent) {
            String action = intent.getAction();
            if (action.equals(Constans.RECEIVER_MESSAGE_NEW)) {
                String stringExtra = intent.getStringExtra("rejectId");
                if (TextUtils.isEmpty(stringExtra)) {
                    return;
                }
                for (String str : stringExtra.split(",")) {
                    if (ControlSponsorActivity.this.mMessage != null && str.equals(ControlSponsorActivity.this.mMessage.getMsgId() + "")) {
                        ControlSponsorActivity.this.closeSocket();
                        ControlSponsorActivity.this.finish();
                    }
                }
                return;
            }
            if (action.equals(SocketManager.RECEIVER_SOCKET_NOTIFY)) {
                int intExtra = intent.getIntExtra(SocketManager.EXTRA_SOCKET_ERROR, -1);
                String stringExtra2 = intent.getStringExtra(SocketManager.EXTRA_SOCKET_DATA);
                long longExtra = intent.getLongExtra(SocketManager.EXTRA_SOCKET_TIME, 0L);
                LogUtil.e("RECEIVER_SOCKET_NOTIFY error:" + intExtra + ", data:" + stringExtra2 + ", remoteTime:" + longExtra);
                if (intExtra != 0 || longExtra <= 0 || ControlSponsorActivity.this.mMessage == null) {
                    return;
                }
                DBMessageUtil.updateDBRemoteTime(ControlSponsorActivity.this, ControlSponsorActivity.this.mMessage.getMsgId() + "", (int) (longExtra / 1000));
            }
        }
    };

    private void checkFloatPermission() {
        if (CheckUtil.checkOp(this, 24) == 0) {
            moveTaskToBack(true);
        } else {
            startActivityForResult(new Intent("android.settings.action.MANAGE_OVERLAY_PERMISSION", Uri.parse("package:" + getPackageName())), 101);
            DialogUtil.dismissDialog();
        }
    }

    private void closeDB() {
        SQLiteDatabase sQLiteDatabase = this.dbRead;
        if (sQLiteDatabase != null) {
            sQLiteDatabase.close();
        }
        SQLiteDatabase sQLiteDatabase2 = this.dbWrite;
        if (sQLiteDatabase2 != null) {
            sQLiteDatabase2.close();
        }
        DatabaseHelper databaseHelper = this.dbHepler;
        if (databaseHelper != null) {
            databaseHelper.close();
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void closeSocket() {
        if (SocketManager.getInstance().getWebSocket() != null) {
            LogUtil.e("WebSocket 发送：关闭");
            SocketManager.getInstance().close();
            SocketManager.getInstance().closeSocket();
        }
    }

    private void init(Bundle bundle) {
        this.baseRequest = new RetrofitRequest();
        this.mGson = new Gson();
        DatabaseHelper databaseHelper = new DatabaseHelper(this);
        this.dbHepler = databaseHelper;
        this.dbWrite = databaseHelper.getWritableDatabase();
        this.dbRead = this.dbHepler.getReadableDatabase();
        if (getIntent() == null) {
            LogUtil.e("getIntent():null");
            finish();
            return;
        }
        this.mUserId = getIntent().getIntExtra(EXTRA_USER_ID, 0);
        LogUtil.e("getIntent() mUserId:" + this.mUserId);
        UserModel selectDBUserForId = selectDBUserForId(this.mUserId);
        this.mUserModel = selectDBUserForId;
        if (selectDBUserForId == null) {
            LogUtil.e("getIntent() mUserModel:null");
            finish();
            return;
        }
        if (BleManager.getInstance(this).getmConnBleDevice() != null) {
            this.mSpeed = BleManager.getInstance(this).getmConnBleDevice().getSpeed();
        }
        if (bundle != null) {
            SocketProductModel socketProductModel = (SocketProductModel) bundle.getSerializable("socketProductModel");
            this.mSocketProductModel = socketProductModel;
            if (socketProductModel != null) {
                initFragment(socketProductModel);
                this.txt_log.setText("已有连接。。。");
            } else {
                SocketManager.getInstance().setOnSocketNotifyListener(this);
                SocketManager.getInstance().connSocket(1);
                this.txt_log.setText("开始连接。。。");
            }
        } else {
            SocketManager.getInstance().setOnSocketNotifyListener(this);
            SocketManager.getInstance().connSocket(1);
            this.txt_log.setText("开始连接。。。");
        }
        registerReceiver();
    }

    private void initFragment(SocketProductModel socketProductModel) {
        ConstraintLayout constraintLayout = this.layout;
        if (constraintLayout != null) {
            constraintLayout.setVisibility(8);
        }
        LinearLayout linearLayout = this.layout_control;
        if (linearLayout != null) {
            linearLayout.setVisibility(0);
        }
        this.mSocketProductModel = socketProductModel;
        FragmentTransaction beginTransaction = getSupportFragmentManager().beginTransaction();
        if (socketProductModel != null) {
            if (!TextUtils.isEmpty(socketProductModel.getType())) {
                if (socketProductModel.getType().equals(this.new_device_type)) {
                    this.mFragment = new PlaySMSFragment();
                } else {
                    this.mFragment = new PlaySMFragment();
                }
                Bundle bundle = new Bundle();
                bundle.putSerializable("ProductModel", socketProductModel);
                this.mFragment.setArguments(bundle);
            } else if (MyApplication.getApplication().getmProductModel() == null || !this.new_device_type.equals(MyApplication.getApplication().getmProductModel().getDeviceType())) {
                this.mFragment = new PlaySMUIFragment(this.mSafeMax);
            } else {
                this.mFragment = new PlaySMSUIFragment(this.mSafeMax);
            }
        }
        beginTransaction.replace(R.id.layout_frame, this.mFragment);
        beginTransaction.commit();
    }

    private void initView() {
        this.title_txt_name = (TextView) findViewById(R.id.title_txt_name);
        this.img_logo = (CircleImageView) findViewById(R.id.img_logo);
        this.layout_refuse = (LinearLayout) findViewById(R.id.layout_refuse);
        this.img_refuse = (ImageView) findViewById(R.id.img_refuse);
        this.txt_refuse = (TextView) findViewById(R.id.txt_refuse);
        this.layout = (ConstraintLayout) findViewById(R.id.layout);
        this.layout_control = (LinearLayout) findViewById(R.id.layout_control);
        this.iplay_layout_bg = (ImageView) findViewById(R.id.iplay_layout_bg);
        this.title_txt_status = (TextView) findViewById(R.id.title_txt_status);
        this.layout_frame = (FrameLayout) findViewById(R.id.layout_frame);
        this.img_small = (ImageView) findViewById(R.id.img_small);
        LanguageUtil.setText(this, this.title_txt_status, "bluetoothDisconnected");
        Glide.with((FragmentActivity) this).load(Integer.valueOf(R.mipmap.bg_iplay)).into(this.iplay_layout_bg);
        this.txt_log = (TextView) findViewById(R.id.txt_log);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void insertDBMessage(MessageModel.Message message) {
        try {
            String obj = SharedPreferencesUtil.get(this, Constans.deleteSession).toString();
            if (!TextUtils.isEmpty(obj) && obj.contains("," + message.getUserId() + ",")) {
                SharedPreferencesUtil.put(this, Constans.deleteSession, obj.replace("," + message.getUserId() + ",", ""));
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        ContentValues contentValues = new ContentValues();
        contentValues.put("msgId", Integer.valueOf(message.getMsgId()));
        contentValues.put("userId", Integer.valueOf(message.getUserId()));
        contentValues.put("sendUserId", Integer.valueOf(MyApplication.getApplication().getmUserModel().getUserId()));
        contentValues.put("type", message.getType());
        contentValues.put("voiceLength", Integer.valueOf(message.getVoiceLength()));
        contentValues.put(FirebaseAnalytics.Param.CONTENT, message.getContent());
        contentValues.put("localPath", message.getLocalPath());
        contentValues.put("date", message.getDate());
        contentValues.put("revoke", Integer.valueOf(message.getRevoke()));
        contentValues.put("androidSend", Integer.valueOf(message.getAndroidSend()));
        contentValues.put("androidRead", (Integer) 1);
        contentValues.put("referenceId", Integer.valueOf(message.getReferenceId()));
        this.dbWrite.replace(DatabaseHelper.TABLE_MESSAGE, null, contentValues);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void insertDBMessageLast(MessageModel.Message message) {
        ContentValues contentValues = new ContentValues();
        contentValues.put("sendUserId", Integer.valueOf(MyApplication.getApplication().getmUserModel().getUserId()));
        contentValues.put("msgLastId", Integer.valueOf(message.getMsgId()));
        this.dbWrite.replace(DatabaseHelper.TABLE_MESSAGE_LAST, null, contentValues);
    }

    private void insertDBReject(String str, String str2) {
        ContentValues contentValues = new ContentValues();
        contentValues.put("reject", str2);
        this.dbWrite.update(DatabaseHelper.TABLE_MESSAGE, contentValues, "msgId = ?", new String[]{str});
        Intent intent = new Intent(Constans.RECEIVER_MESSAGE_NEW);
        intent.putExtra("msgId", str);
        sendBroadcast(intent);
    }

    private void msgAction(int i, int i2) {
        String str;
        String str2 = (String) SharedPreferencesUtil.get(this, Constans.Token);
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
        this.baseRequest.getApi().msgAction(str2, i, str).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<HttpModel>(this, null, false) { // from class: com.hismith.hismithapp.ControlSponsorActivity.4
            @Override // com.hismith.hismithapp.http.RxSubscribe
            protected void onSuccess(HttpModel httpModel) {
                httpModel.getSuccess();
            }
        });
    }

    private void registerReceiver() {
        IntentFilter intentFilter = new IntentFilter();
        intentFilter.addAction(Constans.RECEIVER_MESSAGE_NEW);
        intentFilter.addAction(SocketManager.RECEIVER_SOCKET_NOTIFY);
        AppUtil.registerReceiver(this, this.mReceiver, intentFilter);
    }

    private UserModel selectDBUserForId(int i) {
        Cursor query = this.dbRead.query(DatabaseHelper.TABLE_MESSAGE_USER, null, "userId = ?", new String[]{i + ""}, null, null, null);
        if (query == null || !query.moveToFirst()) {
            return null;
        }
        UserModel userModel = new UserModel();
        userModel.setName(query.getString(query.getColumnIndexOrThrow("name")));
        userModel.setKing(query.getInt(query.getColumnIndexOrThrow("king")));
        if (query.getColumnIndex("gradeIcon") != -1) {
            userModel.setGradeIcon(query.getString(query.getColumnIndexOrThrow("gradeIcon")));
        }
        userModel.setHeadIcon(query.getString(query.getColumnIndexOrThrow("headIcon")));
        userModel.setGender(query.getInt(query.getColumnIndexOrThrow("gender")));
        userModel.setConstellation(query.getString(query.getColumnIndexOrThrow("constellation")));
        query.close();
        return userModel;
    }

    private void setListener() {
        this.img_refuse.setOnClickListener(this);
        this.img_small.setOnClickListener(this);
    }

    public static void startActivity(Context context, int i) {
        Intent intent = new Intent(context, (Class<?>) ControlSponsorActivity.class);
        intent.putExtra(EXTRA_USER_ID, i);
        context.startActivity(intent);
    }

    private void unRegisterReceiver() {
        try {
            unregisterReceiver(this.mReceiver);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override // com.hismith.hismithapp.http.SocketManager.OnSocketNotifyListener
    public void OnNotify(String str) {
        SocketPlayFragment socketPlayFragment;
        if (TextUtils.isEmpty(str)) {
            return;
        }
        try {
            final SocketSpeedModel socketSpeedModel = (SocketSpeedModel) this.mGson.fromJson(str, SocketSpeedModel.class);
            if (socketSpeedModel.getSuccess() == 1) {
                String action = socketSpeedModel.getAction();
                if (!TextUtils.isEmpty(action) && (socketPlayFragment = this.mFragment) != null && socketPlayFragment.isAdded()) {
                    this.mFragment.onNotify(str);
                }
                if (!TextUtils.isEmpty(socketSpeedModel.getGameToken())) {
                    this.txt_log.setText("连接成功，正在通知对方。。。");
                    msgRemoteSend(socketSpeedModel.getGameToken());
                    return;
                }
                boolean z = false;
                if (socketSpeedModel.getClientConnection() == 1) {
                    if (this.mMessage != null) {
                        insertDBReject(this.mMessage.getMsgId() + "", ExifInterface.GPS_MEASUREMENT_3D);
                        SocketManager.getInstance().setMsgId(this.mMessage.getMsgId());
                        SocketManager.getInstance().setRemote_time(System.currentTimeMillis());
                    }
                    SocketProductModel product = socketSpeedModel.getProduct();
                    if (product != null && !TextUtils.isEmpty(product.getType())) {
                        z = true;
                    }
                    initFragment(product);
                    if (z) {
                        ToastUtil.showShortToast(this, LanguageUtil.getLanguageByKey(this, "youCanControl"));
                    } else {
                        ToastUtil.showShortToast(this, LanguageUtil.getLanguageByKey(this, "otherCanControl"));
                    }
                    this.mHandler.removeMessages(1);
                    return;
                }
                if (socketSpeedModel.getOffline() == 1) {
                    closeSocket();
                    finish();
                    return;
                }
                if ("setSpeed".equals(action)) {
                    int speed = socketSpeedModel.getSpeed();
                    this.mSpeed = speed;
                    int i = this.mSafeMax;
                    if (speed > i) {
                        this.mSpeed = i;
                    }
                    BleManager.getInstance(this).writeSpeed(this.mSpeed);
                    return;
                }
                if (!"setMode".equals(action)) {
                    if ("setClimax".equals(action)) {
                        if ("1003".equals(MyApplication.getApplication().getDeviceCode())) {
                            BleManager.getInstance(this).writeOpenClimax();
                            return;
                        } else {
                            BleManager.getInstance(this).writeOpenClimax01();
                            return;
                        }
                    }
                    if ("setStroke".equals(action)) {
                        new Thread(new Runnable() { // from class: com.hismith.hismithapp.ControlSponsorActivity.5
                            @Override // java.lang.Runnable
                            public void run() {
                                try {
                                    BleManager.getInstance(ControlSponsorActivity.this).writeStartPosition(socketSpeedModel.getStrokeMin());
                                    Thread.sleep(100L);
                                    BleManager.getInstance(ControlSponsorActivity.this).writeStopPosition(socketSpeedModel.getStrokeMax());
                                } catch (Exception e) {
                                    e.printStackTrace();
                                }
                            }
                        }).start();
                        return;
                    }
                    if ("setSmooth".equals(action)) {
                        BleManager.getInstance(this).writeSmooth(socketSpeedModel.getSmooth());
                        return;
                    }
                    if (!"setStrokeType".equals(action)) {
                        if ("setStrokeNow".equals(action)) {
                            BleManager.getInstance(this).writePosition(socketSpeedModel.getStrokeNow());
                            return;
                        }
                        return;
                    } else if (socketSpeedModel.getStrokeType() == 1) {
                        BleManager.getInstance(this).writeRunManual();
                        return;
                    } else {
                        BleManager.getInstance(this).writeRunAuto();
                        return;
                    }
                }
                if (socketSpeedModel.isMotor1Mode()) {
                    if (socketSpeedModel.getMode() <= 0) {
                        BleManager.getInstance(this).writeSpeed(this.mSpeed);
                        return;
                    }
                    if (MyApplication.getApplication().getmProductModel() == null || MyApplication.getApplication().getmProductModel().getFuns() == null || MyApplication.getApplication().getmProductModel().getFuns().length <= 0 || MyApplication.getApplication().getmProductModel().getFuns()[0].getModes() == null || MyApplication.getApplication().getmProductModel().getFuns()[0].getModes().length < socketSpeedModel.getMode()) {
                        BleManager.getInstance(this).writeSpeedMode(socketSpeedModel.getMode());
                        return;
                    } else {
                        BleManager.getInstance(this).writeData(MyApplication.getApplication().getmProductModel().getFuns()[0].getModes()[socketSpeedModel.getMode() - 1].getmCommand());
                        return;
                    }
                }
                if (socketSpeedModel.isMotor2Mode()) {
                    if (socketSpeedModel.getMode() == 0) {
                        BleManager.getInstance(this).writeCloseVibrate();
                        return;
                    }
                    if (MyApplication.getApplication().getmProductModel() == null || MyApplication.getApplication().getmProductModel().getFuns() == null || MyApplication.getApplication().getmProductModel().getFuns().length <= 1 || MyApplication.getApplication().getmProductModel().getFuns()[1].getModes() == null || MyApplication.getApplication().getmProductModel().getFuns()[1].getModes().length < socketSpeedModel.getMode()) {
                        BleManager.getInstance(this).writeVibrateMode(socketSpeedModel.getMode());
                    } else {
                        BleManager.getInstance(this).writeData(MyApplication.getApplication().getmProductModel().getFuns()[1].getModes()[socketSpeedModel.getMode() - 1].getmCommand());
                    }
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void msgRemoteSend(String str) {
        String str2 = (String) SharedPreferencesUtil.get(this, Constans.Token);
        String valueOf = String.valueOf(System.currentTimeMillis());
        this.baseRequest.getApi().msgSend(str2, valueOf, Integer.valueOf(this.mUserId), str, 0, Constans.MESSAGE_TYPE_REMOTE).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<MsgSendModel>(this, null, false) { // from class: com.hismith.hismithapp.ControlSponsorActivity.3
            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onComplete() {
                super.onComplete();
                ControlSponsorActivity.this.txt_log.setText("通知已发送，正在对待对方响应");
                ControlSponsorActivity.this.mHandler.sendEmptyMessageDelayed(1, 60000L);
            }

            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onError(Throwable th) {
                super.onError(th);
                ControlSponsorActivity.this.txt_log.setText("网络异常，通知发送失败。请挂断重新发出");
            }

            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(MsgSendModel msgSendModel) {
                if (msgSendModel.getSuccess() != 1) {
                    ControlSponsorActivity.this.img_refuse.setEnabled(false);
                    ControlSponsorActivity.this.mHandler.postDelayed(new Runnable() { // from class: com.hismith.hismithapp.ControlSponsorActivity.3.1
                        @Override // java.lang.Runnable
                        public void run() {
                            ControlSponsorActivity.this.finish();
                        }
                    }, 1500L);
                    return;
                }
                MessageModel messageModel = new MessageModel();
                ControlSponsorActivity controlSponsorActivity = ControlSponsorActivity.this;
                Objects.requireNonNull(messageModel);
                controlSponsorActivity.mMessage = messageModel.new Message();
                ControlSponsorActivity.this.mMessage.setGender(ControlSponsorActivity.this.mUserModel.getGender());
                ControlSponsorActivity.this.mMessage.setHeadIcon(ControlSponsorActivity.this.mUserModel.getHeadIcon());
                ControlSponsorActivity.this.mMessage.setKing(ControlSponsorActivity.this.mUserModel.getKing());
                ControlSponsorActivity.this.mMessage.setGradeIcon(ControlSponsorActivity.this.mUserModel.getGradeIcon());
                ControlSponsorActivity.this.mMessage.setName(ControlSponsorActivity.this.mUserModel.getName());
                ControlSponsorActivity.this.mMessage.setConstellation(ControlSponsorActivity.this.mUserModel.getConstellation());
                ControlSponsorActivity.this.mMessage.setAndroidSend(1);
                ControlSponsorActivity.this.mMessage.setContent(msgSendModel.getContent());
                ControlSponsorActivity.this.mMessage.setDate(msgSendModel.getDate());
                ControlSponsorActivity.this.mMessage.setMsgId(msgSendModel.getMsgId());
                ControlSponsorActivity.this.mMessage.setRevoke(0);
                ControlSponsorActivity.this.mMessage.setType(msgSendModel.getType());
                ControlSponsorActivity.this.mMessage.setUserId(msgSendModel.getUserId());
                ControlSponsorActivity.this.mMessage.setVoiceLength(msgSendModel.getVoiceLength());
                ControlSponsorActivity.this.mMessage.setStatus(0);
                if (msgSendModel.getMsgId() > 0) {
                    ControlSponsorActivity controlSponsorActivity2 = ControlSponsorActivity.this;
                    controlSponsorActivity2.insertDBMessage(controlSponsorActivity2.mMessage);
                }
                ControlSponsorActivity controlSponsorActivity3 = ControlSponsorActivity.this;
                controlSponsorActivity3.insertDBMessageLast(controlSponsorActivity3.mMessage);
                Intent intent = new Intent(Constans.RECEIVER_MESSAGE_NEW);
                intent.putExtra("newMessage", 1);
                ControlSponsorActivity.this.sendBroadcast(intent);
            }
        });
    }

    @Override // androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, android.app.Activity
    protected void onActivityResult(int i, int i2, Intent intent) {
        super.onActivityResult(i, i2, intent);
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        if (view.getId() != R.id.img_refuse) {
            if (view.getId() == R.id.img_small) {
                checkFloatPermission();
            }
        } else {
            this.img_refuse.setEnabled(false);
            if (this.mMessage != null) {
                insertDBReject(this.mMessage.getMsgId() + "", "2");
                msgAction(this.mMessage.getMsgId(), 2);
            }
            closeSocket();
            this.mHandler.postDelayed(new Runnable() { // from class: com.hismith.hismithapp.ControlSponsorActivity.2
                @Override // java.lang.Runnable
                public void run() {
                    ControlSponsorActivity.this.finish();
                }
            }, 1500L);
        }
    }

    @Override // com.hismith.hismithapp.http.SocketManager.OnSocketNotifyListener
    public void onClosed(int i, String str) {
        finish();
    }

    @Override // com.hismith.hismithapp.http.SocketManager.OnSocketNotifyListener
    public void onCodeNull() {
    }

    @Override // com.hismith.hismithapp.bluetooth.BleManager.OnConnectStateListener
    public void onConnected(int i) {
        if (i > 0) {
            runOnUiThread(new Runnable() { // from class: com.hismith.hismithapp.ControlSponsorActivity.7
                @Override // java.lang.Runnable
                public void run() {
                    ControlSponsorActivity.this.title_txt_status.setVisibility(8);
                }
            });
        }
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, androidx.core.app.ComponentActivity, android.app.Activity
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        setContentView(R.layout.activity_control_sponsor);
        initView();
        init(bundle);
        setListener();
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.appcompat.app.AppCompatActivity, androidx.fragment.app.FragmentActivity, android.app.Activity
    protected void onDestroy() {
        super.onDestroy();
        closeDB();
        BleManager.getInstance(this).writeShutdown();
        SocketManager.getInstance().setOnSocketNotifyListener(null);
        unRegisterReceiver();
        this.mHandler.removeCallbacksAndMessages(null);
    }

    @Override // com.hismith.hismithapp.bluetooth.BleManager.OnConnectStateListener
    public void onDisconnected(int i) {
        if (i == 0) {
            runOnUiThread(new Runnable() { // from class: com.hismith.hismithapp.ControlSponsorActivity.8
                @Override // java.lang.Runnable
                public void run() {
                    ControlSponsorActivity.this.title_txt_status.setVisibility(0);
                }
            });
        }
    }

    @Override // com.hismith.hismithapp.http.SocketManager.OnSocketNotifyListener
    public void onFailure(Throwable th, Response response) {
        finish();
    }

    @Override // com.hismith.hismithapp.http.SocketManager.OnSocketNotifyListener
    public void onOpen() {
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.activity.ComponentActivity, androidx.core.app.ComponentActivity, android.app.Activity
    protected void onSaveInstanceState(Bundle bundle) {
        super.onSaveInstanceState(bundle);
        SocketProductModel socketProductModel = this.mSocketProductModel;
        if (socketProductModel != null) {
            bundle.putSerializable("socketProductModel", socketProductModel);
        }
    }
}
