package com.hismith.hismithapp;

import android.content.BroadcastReceiver;
import android.content.ContentValues;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.database.sqlite.SQLiteDatabase;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.text.TextUtils;
import android.view.View;
import android.widget.CompoundButton;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.appcompat.widget.SwitchCompat;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.exifinterface.media.ExifInterface;
import androidx.fragment.app.FragmentActivity;
import androidx.fragment.app.FragmentTransaction;
import androidx.media3.exoplayer.ExoPlayer;
import com.bumptech.glide.Glide;
import com.facebook.internal.NativeProtocol;
import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.bluetooth.BleManager;
import com.hismith.hismithapp.db.DBMessageUtil;
import com.hismith.hismithapp.db.DatabaseHelper;
import com.hismith.hismithapp.fragment.PlaySMFragment;
import com.hismith.hismithapp.fragment.PlaySMSFragment;
import com.hismith.hismithapp.fragment.SocketPlayFragment;
import com.hismith.hismithapp.http.HttpModel;
import com.hismith.hismithapp.http.RetrofitRequest;
import com.hismith.hismithapp.http.RxSubscribe;
import com.hismith.hismithapp.http.SocketManager;
import com.hismith.hismithapp.model.MessageModel;
import com.hismith.hismithapp.model.SocketProductModel;
import com.hismith.hismithapp.model.SocketSpeedModel;
import com.hismith.hismithapp.utils.AppUtil;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import com.hismith.hismithapp.utils.ToastUtil;
import com.hismith.hismithapp.view.CircleImageView;
import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.schedulers.Schedulers;
import okhttp3.Response;

/* loaded from: classes3.dex */
public class ControlAcceptActivity extends BaseActivity implements View.OnClickListener, SocketManager.OnSocketNotifyListener, BleManager.OnConnectStateListener {
    private static String EXTRA_MESSAGE = "extra_message";
    private static final int WHAT_NO_ANSWER = 1;
    private RetrofitRequest baseRequest;
    private DatabaseHelper dbHepler;
    private SQLiteDatabase dbRead;
    private SQLiteDatabase dbWrite;
    private ImageView img_accept;
    private CircleImageView img_logo;
    private ImageView img_refuse;
    private ImageView iplay_layout_bg;
    private ConstraintLayout layout;
    private LinearLayout layout_accept;
    private LinearLayout layout_control;
    private FrameLayout layout_frame;
    private LinearLayout layout_refuse;
    private Gson mGson;
    private MessageModel.Message mMessageModel;
    private SocketProductModel mSocketProductModel;
    private int mSpeed;
    private SwitchCompat switch_hudong;
    private TextView title_txt_name;
    private TextView title_txt_status;
    private TextView txt_accept;
    private TextView txt_refuse;
    private SocketPlayFragment mFragment = null;
    private String new_device_type = Constans.UI_DEVICE_TYPE_CC_SMS;
    private Handler mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.ControlAcceptActivity.1
        @Override // android.os.Handler.Callback
        public boolean handleMessage(Message message) {
            if (message.what != 1) {
                return false;
            }
            ControlAcceptActivity controlAcceptActivity = ControlAcceptActivity.this;
            ToastUtil.showShortToast(controlAcceptActivity, LanguageUtil.getLanguageByKey(controlAcceptActivity, "autoRejected"));
            ControlAcceptActivity.this.closeSocket();
            ControlAcceptActivity.this.finish();
            return false;
        }
    });
    private BroadcastReceiver mReceiver = new BroadcastReceiver() { // from class: com.hismith.hismithapp.ControlAcceptActivity.5
        @Override // android.content.BroadcastReceiver
        public void onReceive(Context context, Intent intent) {
            String action = intent.getAction();
            LogUtil.e("ControlAccept action:" + action);
            if (action.equals(Constans.RECEIVER_MESSAGE_NEW)) {
                String stringExtra = intent.getStringExtra("rejectId");
                String stringExtra2 = intent.getStringExtra("revokeId");
                if (!TextUtils.isEmpty(stringExtra2)) {
                    for (String str : stringExtra2.split(",")) {
                        if (str.equals(ControlAcceptActivity.this.mMessageModel.getMsgId() + "")) {
                            ControlAcceptActivity.this.finish();
                        }
                    }
                }
                TextUtils.isEmpty(stringExtra);
                return;
            }
            if (action.equals(SocketManager.RECEIVER_SOCKET_NOTIFY)) {
                int intExtra = intent.getIntExtra(SocketManager.EXTRA_SOCKET_ERROR, -1);
                String stringExtra3 = intent.getStringExtra(SocketManager.EXTRA_SOCKET_DATA);
                long longExtra = intent.getLongExtra(SocketManager.EXTRA_SOCKET_TIME, 0L);
                LogUtil.e("RECEIVER_SOCKET_NOTIFY error:" + intExtra + ", data:" + stringExtra3 + ", remoteTime:" + longExtra);
                if (intExtra != 0 || longExtra <= 0 || ControlAcceptActivity.this.mMessageModel == null) {
                    return;
                }
                DBMessageUtil.updateDBRemoteTime(ControlAcceptActivity.this, ControlAcceptActivity.this.mMessageModel.getMsgId() + "", (int) (longExtra / 1000));
            }
        }
    };

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
            SocketManager.getInstance().setOnSocketNotifyListener(null);
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
        if (getIntent() != null) {
            this.mMessageModel = (MessageModel.Message) getIntent().getSerializableExtra(EXTRA_MESSAGE);
            LogUtil.e("getIntent() mMessageModel:" + this.mMessageModel);
        } else {
            LogUtil.e("getIntent():null");
        }
        if (this.mMessageModel == null) {
            finish();
        }
        if (BleManager.getInstance(this).getmConnBleDevice() != null) {
            this.mSpeed = BleManager.getInstance(this).getmConnBleDevice().getSpeed();
        }
        if (bundle != null) {
            SocketProductModel socketProductModel = (SocketProductModel) bundle.getSerializable("socketProductModel");
            this.mSocketProductModel = socketProductModel;
            if (socketProductModel != null) {
                initFragment(socketProductModel);
            }
        }
        this.mHandler.sendEmptyMessageDelayed(1, 60000L);
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
            if (TextUtils.isEmpty(socketProductModel.getType()) || !socketProductModel.getType().equals(this.new_device_type)) {
                this.mFragment = new PlaySMFragment();
            } else {
                this.mFragment = new PlaySMSFragment();
            }
            Bundle bundle = new Bundle();
            bundle.putSerializable("ProductModel", socketProductModel);
            this.mFragment.setArguments(bundle);
        }
        beginTransaction.replace(R.id.layout_frame, this.mFragment);
        beginTransaction.commit();
    }

    private void initView() {
        this.title_txt_name = (TextView) findViewById(R.id.title_txt_name);
        this.img_logo = (CircleImageView) findViewById(R.id.img_logo);
        this.layout_refuse = (LinearLayout) findViewById(R.id.layout_refuse);
        this.layout_accept = (LinearLayout) findViewById(R.id.layout_accept);
        this.img_refuse = (ImageView) findViewById(R.id.img_refuse);
        this.img_accept = (ImageView) findViewById(R.id.img_accept);
        this.txt_refuse = (TextView) findViewById(R.id.txt_refuse);
        this.txt_accept = (TextView) findViewById(R.id.txt_accept);
        this.layout = (ConstraintLayout) findViewById(R.id.layout);
        this.layout_control = (LinearLayout) findViewById(R.id.layout_control);
        this.iplay_layout_bg = (ImageView) findViewById(R.id.iplay_layout_bg);
        this.title_txt_status = (TextView) findViewById(R.id.title_txt_status);
        this.layout_frame = (FrameLayout) findViewById(R.id.layout_frame);
        this.switch_hudong = (SwitchCompat) findViewById(R.id.switch_hudong);
        LanguageUtil.setText(this, this.title_txt_status, "bluetoothDisconnected");
        Glide.with((FragmentActivity) this).load(Integer.valueOf(R.mipmap.bg_iplay)).into(this.iplay_layout_bg);
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
        this.baseRequest.getApi().msgAction(str2, i, str).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<HttpModel>(this, null, false) { // from class: com.hismith.hismithapp.ControlAcceptActivity.4
            @Override // com.hismith.hismithapp.http.RxSubscribe
            protected void onSuccess(HttpModel httpModel) {
                if (httpModel.getSuccess() == 1) {
                    ControlAcceptActivity.this.finish();
                }
            }
        });
    }

    private void registerReceiver() {
        IntentFilter intentFilter = new IntentFilter();
        intentFilter.addAction(Constans.RECEIVER_MESSAGE_NEW);
        intentFilter.addAction(SocketManager.RECEIVER_SOCKET_NOTIFY);
        AppUtil.registerReceiver(this, this.mReceiver, intentFilter);
    }

    private void setListener() {
        this.img_refuse.setOnClickListener(this);
        this.img_accept.setOnClickListener(this);
        this.switch_hudong.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() { // from class: com.hismith.hismithapp.ControlAcceptActivity.2
            @Override // android.widget.CompoundButton.OnCheckedChangeListener
            public void onCheckedChanged(CompoundButton compoundButton, boolean z) {
                if (compoundButton.isPressed()) {
                    if (!z) {
                        SocketManager.getInstance().hudongConnection(SocketManager.getInstance().getGameToken(), null);
                    } else if (BleManager.getInstance(ControlAcceptActivity.this).getStatus() == 2) {
                        SocketManager.getInstance().hudongConnection(SocketManager.getInstance().getGameToken(), BleManager.getInstance(ControlAcceptActivity.this).getmConnBleDevice());
                    } else {
                        ControlAcceptActivity.this.switch_hudong.setChecked(false);
                    }
                }
            }
        });
        SocketManager.getInstance().setOnSocketNotifyListener(this);
        BleManager.getInstance(this).setOnConnectStateListener(this);
    }

    public static void startActivity(Context context, MessageModel.Message message) {
        Intent intent = new Intent(context, (Class<?>) ControlAcceptActivity.class);
        intent.putExtra(EXTRA_MESSAGE, message);
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
        SocketProductModel socketProductModel;
        SocketPlayFragment socketPlayFragment;
        SocketPlayFragment socketPlayFragment2;
        String str2 = "";
        if (TextUtils.isEmpty(str)) {
            return;
        }
        try {
            final SocketSpeedModel socketSpeedModel = (SocketSpeedModel) this.mGson.fromJson(str, SocketSpeedModel.class);
            if (socketSpeedModel.getSuccess() == 1) {
                String action = socketSpeedModel.getAction();
                if (!TextUtils.isEmpty(action) && (socketPlayFragment2 = this.mFragment) != null && socketPlayFragment2.isAdded()) {
                    this.mFragment.onNotify(str);
                }
                if (socketSpeedModel.getOffline() == 1) {
                    closeSocket();
                    finish();
                    return;
                }
                if (socketSpeedModel.getServerConnection() == 1) {
                    if (this.mMessageModel != null) {
                        insertDBReject(this.mMessageModel.getMsgId() + "", ExifInterface.GPS_MEASUREMENT_3D);
                        SocketManager.getInstance().setMsgId(this.mMessageModel.getMsgId());
                        SocketManager.getInstance().setRemote_time(System.currentTimeMillis());
                    }
                    SocketProductModel product = socketSpeedModel.getProduct();
                    if (product != null) {
                        initFragment(product);
                    }
                    this.mHandler.removeMessages(1);
                } else if ("setSpeed".equals(action)) {
                    this.mSpeed = socketSpeedModel.getSpeed();
                    BleManager.getInstance(this).writeSpeed(this.mSpeed);
                } else if ("setMode".equals(action)) {
                    if (socketSpeedModel.isMotor1Mode()) {
                        if (socketSpeedModel.getMode() <= 0) {
                            BleManager.getInstance(this).writeSpeed(this.mSpeed);
                        } else if (MyApplication.getApplication().getmProductModel() == null || MyApplication.getApplication().getmProductModel().getFuns() == null || MyApplication.getApplication().getmProductModel().getFuns().length <= 0 || MyApplication.getApplication().getmProductModel().getFuns()[0].getModes() == null || MyApplication.getApplication().getmProductModel().getFuns()[0].getModes().length < socketSpeedModel.getMode()) {
                            BleManager.getInstance(this).writeSpeedMode(socketSpeedModel.getMode());
                        } else {
                            BleManager.getInstance(this).writeData(MyApplication.getApplication().getmProductModel().getFuns()[0].getModes()[socketSpeedModel.getMode() - 1].getmCommand());
                        }
                    } else if (socketSpeedModel.isMotor2Mode()) {
                        if (socketSpeedModel.getMode() == 0) {
                            BleManager.getInstance(this).writeCloseVibrate();
                        } else if (MyApplication.getApplication().getmProductModel() == null || MyApplication.getApplication().getmProductModel().getFuns() == null || MyApplication.getApplication().getmProductModel().getFuns().length <= 1 || MyApplication.getApplication().getmProductModel().getFuns()[1].getModes() == null || MyApplication.getApplication().getmProductModel().getFuns()[1].getModes().length < socketSpeedModel.getMode()) {
                            BleManager.getInstance(this).writeVibrateMode(socketSpeedModel.getMode());
                        } else {
                            BleManager.getInstance(this).writeData(MyApplication.getApplication().getmProductModel().getFuns()[1].getModes()[socketSpeedModel.getMode() - 1].getmCommand());
                        }
                    }
                } else if ("setClimax".equals(action)) {
                    if ("1003".equals(MyApplication.getApplication().getDeviceCode())) {
                        BleManager.getInstance(this).writeOpenClimax();
                    } else {
                        BleManager.getInstance(this).writeOpenClimax01();
                    }
                } else if ("setStroke".equals(action)) {
                    new Thread(new Runnable() { // from class: com.hismith.hismithapp.ControlAcceptActivity.6
                        @Override // java.lang.Runnable
                        public void run() {
                            try {
                                BleManager.getInstance(ControlAcceptActivity.this).writeStartPosition(socketSpeedModel.getStrokeMin());
                                Thread.sleep(100L);
                                BleManager.getInstance(ControlAcceptActivity.this).writeStopPosition(socketSpeedModel.getStrokeMax());
                            } catch (Exception e) {
                                e.printStackTrace();
                            }
                        }
                    }).start();
                } else if ("setSmooth".equals(action)) {
                    BleManager.getInstance(this).writeSmooth(socketSpeedModel.getSmooth());
                } else if ("setStrokeType".equals(action)) {
                    if (socketSpeedModel.getStrokeType() == 1) {
                        BleManager.getInstance(this).writeRunManual();
                    } else {
                        BleManager.getInstance(this).writeRunAuto();
                    }
                } else if ("setStrokeNow".equals(action)) {
                    BleManager.getInstance(this).writePosition(socketSpeedModel.getStrokeNow());
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        try {
            JsonObject asJsonObject = new JsonParser().parse(str).getAsJsonObject();
            if (asJsonObject.get(NativeProtocol.WEB_DIALOG_ACTION) != null) {
                str2 = asJsonObject.get(NativeProtocol.WEB_DIALOG_ACTION).getAsString();
            }
            if (!TextUtils.isEmpty(str2) && (socketPlayFragment = this.mFragment) != null) {
                socketPlayFragment.onNotify(str);
            }
            if (1 != asJsonObject.get("success").getAsInt()) {
                finish();
                return;
            }
            if (asJsonObject.get("offline") != null) {
                if ("1".equals(asJsonObject.get("offline").getAsString())) {
                    finish();
                }
            } else {
                if (asJsonObject.get("serverConnection") == null || asJsonObject.get("serverConnection").getAsInt() != 1 || (socketProductModel = (SocketProductModel) this.mGson.fromJson(asJsonObject.get("product"), SocketProductModel.class)) == null) {
                    return;
                }
                initFragment(socketProductModel);
            }
        } catch (Exception e2) {
            e2.printStackTrace();
        }
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        if (view.getId() == R.id.img_refuse) {
            this.img_refuse.setEnabled(false);
            this.img_accept.setEnabled(false);
            if (this.mMessageModel != null) {
                insertDBReject(this.mMessageModel.getMsgId() + "", "1");
                msgAction(this.mMessageModel.getMsgId(), 4);
            }
            this.mHandler.postDelayed(new Runnable() { // from class: com.hismith.hismithapp.ControlAcceptActivity.3
                @Override // java.lang.Runnable
                public void run() {
                    ControlAcceptActivity.this.finish();
                }
            }, ExoPlayer.DEFAULT_DETACH_SURFACE_TIMEOUT_MS);
            ToastUtil.showShortToast(this, LanguageUtil.getLanguageByKey(this, "remotePlayOver"));
            return;
        }
        if (view.getId() == R.id.img_accept) {
            this.img_refuse.setEnabled(false);
            this.img_accept.setEnabled(false);
            if (this.mMessageModel != null) {
                SocketManager.getInstance().setGameToken(this.mMessageModel.getContent());
                SocketManager.getInstance().connSocket(4);
                this.layout_accept.setVisibility(8);
            }
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
            runOnUiThread(new Runnable() { // from class: com.hismith.hismithapp.ControlAcceptActivity.7
                @Override // java.lang.Runnable
                public void run() {
                    ControlAcceptActivity.this.title_txt_status.setVisibility(8);
                }
            });
        }
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, androidx.core.app.ComponentActivity, android.app.Activity
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        setContentView(R.layout.activity_control_accept);
        initView();
        init(bundle);
        setListener();
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.appcompat.app.AppCompatActivity, androidx.fragment.app.FragmentActivity, android.app.Activity
    protected void onDestroy() {
        super.onDestroy();
        closeDB();
        BleManager.getInstance(this).writeShutdown();
        unRegisterReceiver();
        this.mHandler.removeCallbacksAndMessages(null);
    }

    @Override // com.hismith.hismithapp.bluetooth.BleManager.OnConnectStateListener
    public void onDisconnected(int i) {
        if (i == 0) {
            runOnUiThread(new Runnable() { // from class: com.hismith.hismithapp.ControlAcceptActivity.8
                @Override // java.lang.Runnable
                public void run() {
                    ControlAcceptActivity.this.title_txt_status.setVisibility(0);
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
