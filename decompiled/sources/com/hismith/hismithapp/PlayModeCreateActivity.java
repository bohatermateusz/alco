package com.hismith.hismithapp;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.text.TextUtils;
import android.view.View;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.activity.result.ActivityResult;
import androidx.activity.result.ActivityResultCallback;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.core.content.ContextCompat;
import androidx.core.widget.NestedScrollView;
import androidx.recyclerview.widget.ItemTouchHelper;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.facebook.internal.NativeProtocol;
import com.hismith.hismithapp.adapter.PlayModeCreateAdapter;
import com.hismith.hismithapp.adapter.callback.ItemDragCallback;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.bluetooth.BleManager;
import com.hismith.hismithapp.dialog.PreviewDialog;
import com.hismith.hismithapp.http.RetrofitRequest;
import com.hismith.hismithapp.http.RxSubscribe;
import com.hismith.hismithapp.model.ModeSettingModel;
import com.hismith.hismithapp.model.PlayModeCreateModel;
import com.hismith.hismithapp.model.PlayModeFunItemModel;
import com.hismith.hismithapp.model.PlayModeItemModel;
import com.hismith.hismithapp.model.PlayModeModel;
import com.hismith.hismithapp.product.ProductUtil;
import com.hismith.hismithapp.utils.AppUtil;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.DialogUtil;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import com.hismith.hismithapp.utils.ToastUtil;
import com.hismith.hismithapp.view.plug.PlugSMSPosition;
import com.hismith.hismithapp.view.plug.PlugSMSSmooth;
import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.schedulers.Schedulers;
import java.util.ArrayList;
import java.util.List;

/* loaded from: classes3.dex */
public class PlayModeCreateActivity extends BaseActivity implements View.OnClickListener, PlayModeCreateAdapter.OnDeleteEmptyListener {
    private static final int WHAT_PLAY_MODE_SMOOTH = 5;
    private static final int WHAT_PLAY_MODE_START_PIS = 3;
    private static final int WHAT_PLAY_MODE_STOP_PIS = 4;
    private String action;
    private RetrofitRequest baseRequest;
    private PlayModeCreateAdapter mAdapter;
    private ActivityResultLauncher mBluetoothLauncher;
    private ItemTouchHelper mHelper;
    private ImageView mImgCreate;
    private ImageView mImgPreview;
    private LinearLayout mLayoutComplete;
    private LinearLayout mLayoutCreate;
    private LinearLayout mLayoutLoop;
    private LinearLayoutManager mLayoutManager;
    private LinearLayout mLayoutPlug;
    private LinearLayout mLayoutPopup;
    private LinearLayout mLayoutPreview;
    private LinearLayout mLayoutReversal;
    private LinearLayout mLayoutSingle;
    private List<PlayModeItemModel> mList;
    private RecyclerView mRecyclerView;
    private NestedScrollView mScrollView;
    private ImageView mTitleBack;
    private ImageView mTitleBluetooth;
    private TextView mTitleName;
    private TextView mTxtComplete;
    private TextView mTxtCreate;
    private TextView mTxtLoop;
    private TextView mTxtPreview;
    private TextView mTxtReversal;
    private TextView mTxtSingle;
    private PlugSMSPosition plugSMSPosition;
    private PlugSMSSmooth plugSMSSmooth;
    private PlayModeModel mEditMode = null;
    private int mSMSModeType = 2;
    private int delayMillis = 0;
    Handler mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.PlayModeCreateActivity.1
        @Override // android.os.Handler.Callback
        public boolean handleMessage(Message message) {
            int i = message.what;
            if (i == 3) {
                if (PlayModeCreateActivity.this.mEditMode == null || PlayModeCreateActivity.this.mEditMode.getModeSettingsModel() == null) {
                    return false;
                }
                BleManager.getInstance(PlayModeCreateActivity.this).writeStartPosition(PlayModeCreateActivity.this.mEditMode.getModeSettingsModel().getStrokeMin());
                return false;
            }
            if (i == 4) {
                if (PlayModeCreateActivity.this.mEditMode == null || PlayModeCreateActivity.this.mEditMode.getModeSettingsModel() == null) {
                    return false;
                }
                BleManager.getInstance(PlayModeCreateActivity.this).writeStopPosition(PlayModeCreateActivity.this.mEditMode.getModeSettingsModel().getStrokeMax());
                return false;
            }
            if (i != 5 || PlayModeCreateActivity.this.mEditMode == null || PlayModeCreateActivity.this.mEditMode.getModeSettingsModel() == null) {
                return false;
            }
            BleManager.getInstance(PlayModeCreateActivity.this).writeSmooth(PlayModeCreateActivity.this.mEditMode.getModeSettingsModel().getSmooth());
            return false;
        }
    });
    private BroadcastReceiver mReceiver = new BroadcastReceiver() { // from class: com.hismith.hismithapp.PlayModeCreateActivity.5
        @Override // android.content.BroadcastReceiver
        public void onReceive(Context context, Intent intent) {
            String action = intent.getAction();
            if (action.equals(Constans.RECEIVER_BLE_CONN_STATE)) {
                int intExtra = intent.getIntExtra("ble_state", 0);
                PlayModeCreateActivity playModeCreateActivity = PlayModeCreateActivity.this;
                if (intExtra == 2) {
                    playModeCreateActivity.mTitleBluetooth.setImageDrawable(ContextCompat.getDrawable(PlayModeCreateActivity.this, R.mipmap.ic_play_bluetooth_connect));
                    return;
                } else {
                    playModeCreateActivity.mTitleBluetooth.setImageDrawable(ContextCompat.getDrawable(PlayModeCreateActivity.this, R.mipmap.ic_play_bluetooth_disconnect));
                    return;
                }
            }
            if (action.equals(Constans.RECEIVER_PRODUCT_INFO)) {
                PlayModeCreateActivity.this.initProductData();
                if (PlayModeCreateActivity.this.mAdapter != null) {
                    PlayModeCreateActivity.this.mAdapter.notifyDataSetChanged();
                    return;
                }
                return;
            }
            if (Constans.RECEIVER_BLE_RECE_DATA.equals(action)) {
                String stringExtra = intent.getStringExtra(Constans.BleType);
                String stringExtra2 = intent.getStringExtra(Constans.BleData0);
                String stringExtra3 = intent.getStringExtra(Constans.BleData1);
                if ("CA".equals(stringExtra)) {
                    PlayModeCreateActivity.this.initBleCCData(stringExtra, stringExtra2, stringExtra3);
                }
            }
        }
    };

    class OnCycleClickListener implements View.OnClickListener {
        OnCycleClickListener() {
        }

        @Override // android.view.View.OnClickListener
        public void onClick(View view) {
            if (PlayModeCreateActivity.this.mList.size() == 0) {
                return;
            }
            if (PlayModeCreateActivity.this.mLayoutPopup.getVisibility() == 0) {
                PlayModeCreateActivity.this.mLayoutPopup.setVisibility(4);
            }
            PlayModeItemModel playModeItemModel = new PlayModeItemModel("0");
            playModeItemModel.setStart(1);
            playModeItemModel.setEnd(PlayModeCreateActivity.this.mAdapter.getmSerialId());
            playModeItemModel.setCount(1);
            PlayModeCreateActivity.this.mAdapter.addCycle(playModeItemModel);
            PlayModeCreateActivity.this.mAdapter.notifyDataSetChanged();
            PlayModeCreateActivity.this.mRecyclerView.scrollToPosition(PlayModeCreateActivity.this.mAdapter.getItemCount() - 1);
        }
    }

    class OnReversalClickListener implements View.OnClickListener {
        OnReversalClickListener() {
        }

        @Override // android.view.View.OnClickListener
        public void onClick(View view) {
            if (PlayModeCreateActivity.this.mList.size() == 0) {
                PlayModeCreateActivity.this.hasData();
            }
            if (PlayModeCreateActivity.this.mLayoutPopup.getVisibility() == 0) {
                PlayModeCreateActivity.this.mLayoutPopup.setVisibility(4);
            }
            PlayModeItemModel playModeItemModel = new PlayModeItemModel((PlayModeCreateActivity.this.mSMSModeType == 2 && ProductUtil.isSMSType()) ? "S" : ProductUtil.isSMVType() ? "2" : "1");
            playModeItemModel.setStart(255);
            PlayModeCreateActivity.this.mAdapter.addSingle(playModeItemModel);
            PlayModeCreateActivity.this.mAdapter.notifyDataSetChanged();
            PlayModeCreateActivity.this.mRecyclerView.scrollToPosition(PlayModeCreateActivity.this.mAdapter.getItemCount() - 1);
        }
    }

    class OnSingleClickListener implements View.OnClickListener {
        OnSingleClickListener() {
        }

        @Override // android.view.View.OnClickListener
        public void onClick(View view) {
            if (PlayModeCreateActivity.this.mList.size() == 0) {
                PlayModeCreateActivity.this.hasData();
            }
            if (PlayModeCreateActivity.this.mLayoutPopup.getVisibility() == 0) {
                PlayModeCreateActivity.this.mLayoutPopup.setVisibility(4);
            }
            PlayModeItemModel playModeItemModel = new PlayModeItemModel((PlayModeCreateActivity.this.mSMSModeType == 2 && ProductUtil.isSMSType()) ? "S" : ProductUtil.isSMVType() ? "2" : "1");
            if (PlayModeCreateActivity.this.mSMSModeType == 2 && ProductUtil.isSMSType()) {
                playModeItemModel.setEnd(100);
            }
            PlayModeCreateActivity.this.mAdapter.addSingle(playModeItemModel);
            PlayModeCreateActivity.this.mAdapter.notifyDataSetChanged();
            PlayModeCreateActivity.this.mRecyclerView.scrollToPosition(PlayModeCreateActivity.this.mAdapter.getItemCount() - 1);
        }
    }

    private boolean bluetoothIsEnabled() {
        if (Build.VERSION.SDK_INT >= 31) {
            return true;
        }
        if (BleManager.getInstance(this).getBluetoothAdapter() == null) {
            return false;
        }
        if (BleManager.getInstance(this).getBluetoothAdapter().isEnabled()) {
            return true;
        }
        Intent intent = new Intent("android.bluetooth.adapter.action.REQUEST_ENABLE");
        ActivityResultLauncher activityResultLauncher = this.mBluetoothLauncher;
        if (activityResultLauncher == null) {
            return false;
        }
        activityResultLauncher.launch(intent);
        return false;
    }

    private String getModeData() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < this.mList.size(); i++) {
            String single = this.mList.get(i).getSingle();
            char c = (ProductUtil.isSMSType() && "1".equals(single)) ? (char) 2 : (char) 1;
            sb.append(single);
            if ("1".equals(single) && c == 2 && i > 0) {
                int i2 = i - 1;
                while (true) {
                    if (i2 < 0) {
                        break;
                    }
                    if ("1".equals(this.mList.get(i2).getSingle())) {
                        int end = this.mList.get(i2).getEnd();
                        List<PlayModeItemModel> list = this.mList;
                        if (end < 16) {
                            sb.append("0" + Integer.toHexString(list.get(i2).getEnd()).toUpperCase());
                        } else {
                            sb.append(Integer.toHexString(list.get(i2).getEnd()).toUpperCase());
                        }
                    } else {
                        i2--;
                    }
                }
            } else {
                int start = this.mList.get(i).getStart();
                List<PlayModeItemModel> list2 = this.mList;
                if (start < 16) {
                    sb.append("0" + Integer.toHexString(list2.get(i).getStart()).toUpperCase());
                } else {
                    sb.append(Integer.toHexString(list2.get(i).getStart()).toUpperCase());
                }
            }
            int end2 = this.mList.get(i).getEnd();
            List<PlayModeItemModel> list3 = this.mList;
            if (end2 < 16) {
                sb.append("0" + Integer.toHexString(list3.get(i).getEnd()).toUpperCase());
            } else {
                sb.append(Integer.toHexString(list3.get(i).getEnd()).toUpperCase());
            }
            if ("2".equals(single) || "S".equals(single)) {
                int start2 = this.mList.get(i).getStart2();
                List<PlayModeItemModel> list4 = this.mList;
                if (start2 < 16) {
                    sb.append("0" + Integer.toHexString(list4.get(i).getStart2()).toUpperCase());
                } else {
                    sb.append(Integer.toHexString(list4.get(i).getStart2()).toUpperCase());
                }
                int end22 = this.mList.get(i).getEnd2();
                List<PlayModeItemModel> list5 = this.mList;
                if (end22 < 16) {
                    sb.append("0" + Integer.toHexString(list5.get(i).getEnd2()).toUpperCase());
                } else {
                    sb.append(Integer.toHexString(list5.get(i).getEnd2()).toUpperCase());
                }
            }
            if ("S".equals(single)) {
                sb.append(Integer.toHexString(this.mList.get(i).getSmooth()).toUpperCase());
            }
            if ("1".equals(single) || "2".equals(single) || "S".equals(single)) {
                int seconds = this.mList.get(i).getSeconds();
                List<PlayModeItemModel> list6 = this.mList;
                if (seconds < 16) {
                    sb.append("0" + Integer.toHexString(list6.get(i).getSeconds()).toUpperCase());
                } else {
                    sb.append(Integer.toHexString(list6.get(i).getSeconds()).toUpperCase());
                }
                sb.append(Integer.toHexString(this.mList.get(i).getMilli()).toUpperCase());
            } else if ("0".equals(single)) {
                int count = this.mList.get(i).getCount();
                List<PlayModeItemModel> list7 = this.mList;
                if (count < 16) {
                    sb.append("0" + Integer.toHexString(list7.get(i).getCount()).toUpperCase());
                } else {
                    sb.append(Integer.toHexString(list7.get(i).getCount()).toUpperCase());
                }
            }
            sb.append("G");
        }
        return sb.toString();
    }

    private String getVCModeData() {
        StringBuilder sb = new StringBuilder("VC01T");
        sb.append(Integer.toHexString(100)).append(Constans.DEVICE_TYPE_FUN);
        ArrayList arrayList = new ArrayList();
        if (BleManager.getInstance(this).hasPiuFunBle()) {
            arrayList.add(PlayModeFunItemModel.FUN_PIU);
        }
        if (!arrayList.isEmpty()) {
            sb.append(String.join(",", arrayList));
        }
        for (int i = 0; i < this.mList.size(); i++) {
        }
        return sb.toString();
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void hasData() {
        this.mTxtPreview.setTextColor(getResources().getColor(R.color.play_mode_create_preview_enable_text_color));
        this.mTxtComplete.setTextColor(getResources().getColor(R.color.play_mode_create_enable_text_color));
        this.mLayoutPreview.setOnClickListener(this);
        this.mLayoutComplete.setOnClickListener(this);
    }

    private void init() {
        this.baseRequest = new RetrofitRequest();
        int status = BleManager.getInstance(this).getStatus();
        ImageView imageView = this.mTitleBluetooth;
        if (status == 2) {
            imageView.setImageDrawable(ContextCompat.getDrawable(this, R.mipmap.ic_play_bluetooth_connect));
        } else {
            imageView.setImageDrawable(ContextCompat.getDrawable(this, R.mipmap.ic_play_bluetooth_disconnect));
        }
        LinearLayoutManager linearLayoutManager = new LinearLayoutManager(this);
        this.mLayoutManager = linearLayoutManager;
        this.mRecyclerView.setLayoutManager(linearLayoutManager);
        this.mList = new ArrayList();
        PlayModeCreateAdapter playModeCreateAdapter = new PlayModeCreateAdapter(this, this.mList, this);
        this.mAdapter = playModeCreateAdapter;
        this.mRecyclerView.setAdapter(playModeCreateAdapter);
        ItemTouchHelper itemTouchHelper = new ItemTouchHelper(new ItemDragCallback(this, this.mAdapter, this.mList));
        this.mHelper = itemTouchHelper;
        itemTouchHelper.attachToRecyclerView(this.mRecyclerView);
        initActivityResult();
        makeFilter();
        initEditMode();
        initProductData();
    }

    private void initActivityResult() {
        this.mBluetoothLauncher = registerForActivityResult(new ActivityResultContracts.StartActivityForResult(), new ActivityResultCallback<ActivityResult>() { // from class: com.hismith.hismithapp.PlayModeCreateActivity.2
            @Override // androidx.activity.result.ActivityResultCallback
            public void onActivityResult(ActivityResult activityResult) {
                if (activityResult.getResultCode() == -1) {
                    LogUtil.e("蓝牙打开");
                    PlayModeCreateActivity.this.startActivity(new Intent(PlayModeCreateActivity.this, (Class<?>) BluetootListActivity.class));
                } else if (activityResult.getResultCode() == 0) {
                    LogUtil.e("蓝牙打不开");
                }
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void initBleCCData(String str, String str2, String str3) {
        PlugSMSSmooth plugSMSSmooth;
        int intValue = Integer.valueOf(str2, 16).intValue();
        int intValue2 = Integer.valueOf(str3, 16).intValue();
        if (intValue == 7) {
            PlugSMSPosition plugSMSPosition = this.plugSMSPosition;
            if (plugSMSPosition != null) {
                plugSMSPosition.setmPositionStart(intValue2);
                return;
            }
            return;
        }
        if (intValue == 8) {
            PlugSMSPosition plugSMSPosition2 = this.plugSMSPosition;
            if (plugSMSPosition2 != null) {
                plugSMSPosition2.setmPositionEnd(intValue2);
                return;
            }
            return;
        }
        if (intValue != 9 || (plugSMSSmooth = this.plugSMSSmooth) == null) {
            return;
        }
        plugSMSSmooth.setSmooth(intValue2);
    }

    private void initEditMode() {
        try {
            if (getIntent() != null) {
                this.action = getIntent().getStringExtra(NativeProtocol.WEB_DIALOG_ACTION);
                PlayModeModel playModeModel = (PlayModeModel) getIntent().getParcelableExtra("editMode");
                this.mEditMode = playModeModel;
                if (playModeModel != null) {
                    String[] split = playModeModel.getModeData().toUpperCase().split("G");
                    int i = 0;
                    int i2 = 0;
                    int i3 = 0;
                    while (i2 < split.length) {
                        String substring = split[i2].substring(i, 1);
                        if (i2 == 0 && "1".equals(substring) && ProductUtil.isSMSType()) {
                            this.mSMSModeType = 1;
                        } else if (i2 == 0 && "S".equals(substring)) {
                            this.mSMSModeType = 2;
                        }
                        PlayModeItemModel playModeItemModel = new PlayModeItemModel(substring);
                        if ("2".equals(substring)) {
                            int intValue = Integer.valueOf(split[i2].substring(1, 3), 16).intValue();
                            int intValue2 = Integer.valueOf(split[i2].substring(3, 5), 16).intValue();
                            int intValue3 = Integer.valueOf(split[i2].substring(5, 7), 16).intValue();
                            int intValue4 = Integer.valueOf(split[i2].substring(7, 9), 16).intValue();
                            int intValue5 = Integer.valueOf(split[i2].substring(9, 11), 16).intValue();
                            int intValue6 = Integer.valueOf(split[i2].substring(11, 12), 16).intValue();
                            playModeItemModel.setStart(intValue);
                            playModeItemModel.setEnd(intValue2);
                            playModeItemModel.setStart2(intValue3);
                            playModeItemModel.setEnd2(intValue4);
                            playModeItemModel.setSeconds(intValue5);
                            playModeItemModel.setMilli(intValue6);
                            i3++;
                            playModeItemModel.setSerialId(i3);
                            this.mAdapter.addSingle(playModeItemModel);
                        } else if ("1".equals(substring)) {
                            int intValue7 = Integer.valueOf(split[i2].substring(1, 3), 16).intValue();
                            int intValue8 = Integer.valueOf(split[i2].substring(3, 5), 16).intValue();
                            int intValue9 = Integer.valueOf(split[i2].substring(5, 7), 16).intValue();
                            int intValue10 = Integer.valueOf(split[i2].substring(7, 8), 16).intValue();
                            playModeItemModel.setStart(intValue7);
                            playModeItemModel.setEnd(intValue8);
                            playModeItemModel.setSeconds(intValue9);
                            playModeItemModel.setMilli(intValue10);
                            i3++;
                            playModeItemModel.setSerialId(i3);
                            this.mAdapter.addSingle(playModeItemModel);
                        } else if ("0".equals(substring)) {
                            int intValue11 = Integer.valueOf(split[i2].substring(1, 3), 16).intValue();
                            int intValue12 = Integer.valueOf(split[i2].substring(3, 5), 16).intValue();
                            int intValue13 = Integer.valueOf(split[i2].substring(5, 7), 16).intValue();
                            playModeItemModel.setStart(intValue11);
                            playModeItemModel.setEnd(intValue12);
                            playModeItemModel.setCount(intValue13);
                            this.mAdapter.addCycle(playModeItemModel);
                        } else if ("S".equals(substring)) {
                            int intValue14 = Integer.valueOf(split[i2].substring(1, 3), 16).intValue();
                            int intValue15 = Integer.valueOf(split[i2].substring(3, 5), 16).intValue();
                            int intValue16 = Integer.valueOf(split[i2].substring(5, 7), 16).intValue();
                            int intValue17 = Integer.valueOf(split[i2].substring(7, 9), 16).intValue();
                            int intValue18 = Integer.valueOf(split[i2].substring(9, 10), 16).intValue();
                            int intValue19 = Integer.valueOf(split[i2].substring(10, 12), 16).intValue();
                            int intValue20 = Integer.valueOf(split[i2].substring(12, 13), 16).intValue();
                            playModeItemModel.setStart(intValue14);
                            playModeItemModel.setEnd(intValue15);
                            playModeItemModel.setStart2(intValue16);
                            playModeItemModel.setEnd2(intValue17);
                            playModeItemModel.setSmooth(intValue18);
                            playModeItemModel.setSeconds(intValue19);
                            playModeItemModel.setMilli(intValue20);
                            i3++;
                            playModeItemModel.setSerialId(i3);
                            LogUtil.e("model:" + playModeItemModel.toString());
                            this.mAdapter.addSingle(playModeItemModel);
                        }
                        this.mAdapter.notifyDataSetChanged();
                        i2++;
                        i = 0;
                    }
                    if (this.mList.size() > 0) {
                        hasData();
                    }
                    if (TextUtils.isEmpty(this.mEditMode.getModeSetting())) {
                        return;
                    }
                    this.mEditMode.initModeSettingsModel();
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void initProductData() {
        int i;
        int i2;
        int smoothness;
        this.delayMillis = 0;
        if (ProductUtil.isSMSType() && this.mSMSModeType == 1) {
            PlayModeModel playModeModel = this.mEditMode;
            if (playModeModel != null && playModeModel.getModeSettingsModel() != null) {
                i = this.mEditMode.getModeSettingsModel().getStrokeMin();
                i2 = this.mEditMode.getModeSettingsModel().getStrokeMax();
                this.mHandler.sendEmptyMessageDelayed(3, this.delayMillis);
                int i3 = this.delayMillis + 200;
                this.delayMillis = i3;
                this.mHandler.sendEmptyMessageDelayed(4, i3);
                this.delayMillis += 200;
            } else if (BleManager.getInstance(this).getmConnBleDevice() != null) {
                i = BleManager.getInstance(this).getmConnBleDevice().getStartPosition();
                i2 = BleManager.getInstance(this).getmConnBleDevice().getStopPosition();
            } else {
                i = 0;
                i2 = 0;
            }
            if (this.plugSMSPosition == null) {
                PlugSMSPosition plugSMSPosition = new PlugSMSPosition(this);
                this.plugSMSPosition = plugSMSPosition;
                this.mLayoutPlug.addView(plugSMSPosition.getBaseView());
            }
            this.plugSMSPosition.setmPositionStart(i);
            if (Math.max(i2, i + Constans.start_stop_position_diff) > 100) {
                i2 = 100;
            }
            this.plugSMSPosition.setmPositionEnd(i2);
            PlayModeModel playModeModel2 = this.mEditMode;
            if (playModeModel2 == null || playModeModel2.getModeSettingsModel() == null) {
                smoothness = BleManager.getInstance(this).getmConnBleDevice() != null ? BleManager.getInstance(this).getmConnBleDevice().getSmoothness() : 0;
            } else {
                smoothness = this.mEditMode.getModeSettingsModel().getSmooth();
                this.mHandler.sendEmptyMessageDelayed(5, this.delayMillis);
                this.delayMillis += 200;
            }
            if (this.plugSMSSmooth == null) {
                PlugSMSSmooth plugSMSSmooth = new PlugSMSSmooth(this, this.mLayoutPlug);
                this.plugSMSSmooth = plugSMSSmooth;
                this.mLayoutPlug.addView(plugSMSSmooth.getBaseView());
            }
            this.plugSMSSmooth.setSmooth(smoothness);
            this.mLayoutPlug.setVisibility(0);
        } else {
            this.mLayoutPlug.removeAllViews();
            this.mLayoutPlug.setVisibility(8);
            this.plugSMSPosition = null;
            this.plugSMSSmooth = null;
        }
        boolean hasReversal = ProductUtil.hasReversal();
        LinearLayout linearLayout = this.mLayoutReversal;
        if (hasReversal) {
            linearLayout.setVisibility(0);
        } else {
            linearLayout.setVisibility(8);
        }
        for (int size = this.mList.size() - 1; size >= 0; size--) {
            PlayModeItemModel playModeItemModel = this.mList.get(size);
            if (ProductUtil.isSMSType()) {
                if (this.mSMSModeType == 2) {
                    if (!"0".equals(playModeItemModel.getSingle())) {
                        playModeItemModel.setSingle("S");
                    }
                } else if (!"0".equals(playModeItemModel.getSingle())) {
                    playModeItemModel.setSingle("1");
                }
            } else if (ProductUtil.isSMVType()) {
                if (!"0".equals(playModeItemModel.getSingle())) {
                    playModeItemModel.setSingle("2");
                }
            } else if (!"0".equals(playModeItemModel.getSingle())) {
                playModeItemModel.setSingle("1");
            }
        }
    }

    private void initView() {
        this.mTitleBack = (ImageView) findViewById(R.id.title_img_back);
        this.mTitleName = (TextView) findViewById(R.id.title_txt_name);
        this.mTitleBluetooth = (ImageView) findViewById(R.id.title_img_bluetooth);
        this.mScrollView = (NestedScrollView) findViewById(R.id.scroll_view);
        this.mRecyclerView = (RecyclerView) findViewById(R.id.mode_create_recycler_view);
        this.mImgCreate = (ImageView) findViewById(R.id.mode_create_img_create);
        this.mImgPreview = (ImageView) findViewById(R.id.mode_create_img_preview);
        this.mTxtCreate = (TextView) findViewById(R.id.mode_create_txt_create);
        this.mTxtPreview = (TextView) findViewById(R.id.mode_create_txt_preview);
        this.mTxtComplete = (TextView) findViewById(R.id.mode_create_txt_complete);
        this.mLayoutCreate = (LinearLayout) findViewById(R.id.mode_create_layout_create);
        this.mLayoutPreview = (LinearLayout) findViewById(R.id.mode_create_layout_preview);
        this.mLayoutComplete = (LinearLayout) findViewById(R.id.mode_create_layout_complete);
        this.mLayoutPopup = (LinearLayout) findViewById(R.id.mode_create_layout_popup);
        this.mLayoutSingle = (LinearLayout) findViewById(R.id.layout_single);
        this.mLayoutReversal = (LinearLayout) findViewById(R.id.layout_reversal);
        this.mLayoutLoop = (LinearLayout) findViewById(R.id.layout_loop);
        this.mTxtSingle = (TextView) findViewById(R.id.txt_single);
        this.mTxtReversal = (TextView) findViewById(R.id.txt_reversal);
        this.mTxtLoop = (TextView) findViewById(R.id.txt_loop);
        this.mLayoutPlug = (LinearLayout) findViewById(R.id.layout_plug);
        LanguageUtil.setText(this, this.mTitleName, "addMode");
        LanguageUtil.setText(this, this.mTxtCreate, "add");
        LanguageUtil.setText(this, this.mTxtPreview, "preview");
        LanguageUtil.setText(this, this.mTxtComplete, "complete");
        LanguageUtil.setText(this, this.mTxtSingle, "addSingle");
        LanguageUtil.setText(this, this.mTxtReversal, "rotate");
        LanguageUtil.setText(this, this.mTxtLoop, "addCycle");
    }

    private boolean isNotNullData() {
        boolean z = true;
        for (int i = 0; i < this.mList.size(); i++) {
            try {
                if ("2".equals(this.mList.get(i).getSingle())) {
                    Integer valueOf = Integer.valueOf(this.mList.get(i).getStart());
                    Integer valueOf2 = Integer.valueOf(this.mList.get(i).getEnd());
                    Integer valueOf3 = Integer.valueOf(this.mList.get(i).getStart2());
                    Integer valueOf4 = Integer.valueOf(this.mList.get(i).getEnd2());
                    Integer valueOf5 = Integer.valueOf(this.mList.get(i).getSeconds());
                    Integer valueOf6 = Integer.valueOf(this.mList.get(i).getMilli());
                    if (valueOf != null && valueOf2 != null && valueOf3 != null && valueOf4 != null && valueOf5 != null && valueOf6 != null) {
                        if (this.mList.get(i).isError()) {
                            this.mList.get(i).setError(false);
                            this.mAdapter.notifyItemChanged(i, "");
                        }
                    }
                    this.mList.get(i).setError(true);
                    this.mAdapter.notifyItemChanged(i, "");
                    return false;
                }
                if ("1".equals(this.mList.get(i).getSingle())) {
                    Integer valueOf7 = Integer.valueOf(this.mList.get(i).getStart());
                    Integer valueOf8 = Integer.valueOf(this.mList.get(i).getEnd());
                    Integer valueOf9 = Integer.valueOf(this.mList.get(i).getSeconds());
                    Integer valueOf10 = Integer.valueOf(this.mList.get(i).getMilli());
                    if (valueOf7 != null && valueOf8 != null && valueOf9 != null && valueOf10 != null) {
                        if (this.mList.get(i).isError()) {
                            this.mList.get(i).setError(false);
                            this.mAdapter.notifyItemChanged(i, "");
                        }
                    }
                    this.mList.get(i).setError(true);
                    this.mAdapter.notifyItemChanged(i, "");
                    return false;
                }
                if ("0".equals(this.mList.get(i).getSingle())) {
                    Integer valueOf11 = Integer.valueOf(this.mList.get(i).getStart());
                    Integer valueOf12 = Integer.valueOf(this.mList.get(i).getEnd());
                    Integer valueOf13 = Integer.valueOf(this.mList.get(i).getCount());
                    if (valueOf11 != null && valueOf12 != null && valueOf13 != null) {
                        if (this.mList.get(i).isError()) {
                            this.mList.get(i).setError(false);
                            this.mAdapter.notifyItemChanged(i, "");
                        }
                    }
                    this.mList.get(i).setError(true);
                    this.mAdapter.notifyItemChanged(i, "");
                    return false;
                }
                "S".equals(this.mList.get(i).getSingle());
                this.mList.get(i).setError(false);
            } catch (Exception e) {
                e.printStackTrace();
                z = false;
            }
        }
        return z;
    }

    private void makeFilter() {
        IntentFilter intentFilter = new IntentFilter();
        intentFilter.addAction(Constans.RECEIVER_BLE_CONN_STATE);
        intentFilter.addAction(Constans.RECEIVER_PRODUCT_INFO);
        intentFilter.addAction(Constans.RECEIVER_BLE_RECE_DATA);
        AppUtil.registerReceiver(this, this.mReceiver, intentFilter);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void playModeSave(String str, String str2, String str3) {
        String str4;
        String str5 = (String) SharedPreferencesUtil.get(this, Constans.Token);
        String deviceCode = MyApplication.getApplication().getDeviceCode();
        if (TextUtils.isEmpty(deviceCode)) {
            ToastUtil.showShortToast(this, LanguageUtil.getLanguageByKey(this, "deviceNotConnect"));
            return;
        }
        if (!ProductUtil.hasSType()) {
            ToastUtil.showShortToast(this, LanguageUtil.getLanguageByKey(this, "deviceNotCreateMode"));
            return;
        }
        boolean z = true;
        if (ProductUtil.isSMSType() && this.mSMSModeType == 1 && BleManager.getInstance(this).getmConnBleDevice() != null) {
            try {
                str4 = this.gson.toJson(new ModeSettingModel(BleManager.getInstance(this).getmConnBleDevice().getStartPosition(), BleManager.getInstance(this).getmConnBleDevice().getStopPosition(), BleManager.getInstance(this).getmConnBleDevice().getSmoothness()));
            } catch (Exception e) {
                e.printStackTrace();
            }
            String str6 = str4;
            LogUtil.e("setting:" + str6);
            this.baseRequest.getApi().playModeSave(str5, str, str2, deviceCode, str6, str3).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<PlayModeCreateModel>(this, this.progressDialog, z) { // from class: com.hismith.hismithapp.PlayModeCreateActivity.3
                /* JADX INFO: Access modifiers changed from: protected */
                @Override // com.hismith.hismithapp.http.RxSubscribe
                public void onSuccess(PlayModeCreateModel playModeCreateModel) {
                    if (playModeCreateModel.getSuccess() == 1) {
                        ToastUtil.showShortToastCenter(PlayModeCreateActivity.this, LanguageUtil.getLanguageByKey(PlayModeCreateActivity.this, "saveed"));
                        Intent intent = new Intent();
                        intent.putExtra("modeData", playModeCreateModel.getModeData());
                        intent.putExtra("modePoint", playModeCreateModel.getModePoints());
                        intent.putExtra("modeSetting", playModeCreateModel.getModeSetting());
                        PlayModeCreateActivity.this.setResult(17, intent);
                        PlayModeCreateActivity.this.finish();
                    }
                }
            });
        }
        str4 = null;
        String str62 = str4;
        LogUtil.e("setting:" + str62);
        this.baseRequest.getApi().playModeSave(str5, str, str2, deviceCode, str62, str3).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<PlayModeCreateModel>(this, this.progressDialog, z) { // from class: com.hismith.hismithapp.PlayModeCreateActivity.3
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(PlayModeCreateModel playModeCreateModel) {
                if (playModeCreateModel.getSuccess() == 1) {
                    ToastUtil.showShortToastCenter(PlayModeCreateActivity.this, LanguageUtil.getLanguageByKey(PlayModeCreateActivity.this, "saveed"));
                    Intent intent = new Intent();
                    intent.putExtra("modeData", playModeCreateModel.getModeData());
                    intent.putExtra("modePoint", playModeCreateModel.getModePoints());
                    intent.putExtra("modeSetting", playModeCreateModel.getModeSetting());
                    PlayModeCreateActivity.this.setResult(17, intent);
                    PlayModeCreateActivity.this.finish();
                }
            }
        });
    }

    private void setListener() {
        this.mTitleBack.setOnClickListener(this);
        this.mTitleBluetooth.setOnClickListener(this);
        this.mLayoutCreate.setOnClickListener(this);
        this.mLayoutSingle.setOnClickListener(new OnSingleClickListener());
        this.mLayoutReversal.setOnClickListener(new OnReversalClickListener());
        this.mLayoutLoop.setOnClickListener(new OnCycleClickListener());
    }

    private void unRegisterReceiver() {
        try {
            unregisterReceiver(this.mReceiver);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        int id = view.getId();
        if (id == R.id.title_img_back) {
            finish();
            return;
        }
        if (id == R.id.title_img_bluetooth) {
            if (bluetoothIsEnabled()) {
                startActivity(new Intent(this, (Class<?>) BluetootListActivity.class));
                return;
            }
            return;
        }
        if (id == R.id.mode_create_layout_create) {
            int visibility = this.mLayoutPopup.getVisibility();
            LinearLayout linearLayout = this.mLayoutPopup;
            if (visibility == 0) {
                linearLayout.setVisibility(4);
                return;
            } else {
                linearLayout.setVisibility(0);
                return;
            }
        }
        if (id == R.id.mode_create_layout_preview) {
            if (isNotNullData()) {
                if (BleManager.getInstance(this).getStatus() != 2) {
                    ToastUtil.showShortToastCenter(this, LanguageUtil.getLanguageByKey(this, "noConnectedDevice"));
                    return;
                }
                String modeData = getModeData();
                LogUtil.e("modeData:" + modeData);
                new PreviewDialog(this, modeData).show();
                return;
            }
            return;
        }
        if (id == R.id.mode_create_layout_complete && isNotNullData()) {
            final String modeData2 = getModeData();
            LogUtil.e("modeData:" + modeData2);
            if (this.mEditMode == null || !"edit".equals(this.action)) {
                DialogUtil.showEditDialog2(this, LanguageUtil.getLanguageByKey(this, "modeName"), "", null, null, new DialogUtil.OnEditContentCallback() { // from class: com.hismith.hismithapp.PlayModeCreateActivity.4
                    @Override // com.hismith.hismithapp.utils.DialogUtil.OnEditContentCallback
                    public void onEditContent(String str) {
                        boolean isEmpty = TextUtils.isEmpty(str);
                        PlayModeCreateActivity playModeCreateActivity = PlayModeCreateActivity.this;
                        if (isEmpty) {
                            ToastUtil.showShortToastCenter(playModeCreateActivity, LanguageUtil.getLanguageByKey(playModeCreateActivity, "pleaseEnterModeName"));
                        } else {
                            playModeCreateActivity.playModeSave(null, str, modeData2);
                        }
                    }
                });
            } else {
                playModeSave(this.mEditMode.getModeId() + "", this.mEditMode.getModeName(), modeData2);
            }
        }
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, androidx.core.app.ComponentActivity, android.app.Activity
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        setContentView(R.layout.activity_play_mode_create);
        initView();
        init();
        setListener();
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.appcompat.app.AppCompatActivity, androidx.fragment.app.FragmentActivity, android.app.Activity
    protected void onDestroy() {
        super.onDestroy();
        unRegisterReceiver();
    }

    @Override // com.hismith.hismithapp.adapter.PlayModeCreateAdapter.OnDeleteEmptyListener
    public void onEmpty() {
        this.mTxtPreview.setTextColor(getResources().getColor(R.color.play_mode_create_preview_disable_text_color));
        this.mTxtComplete.setTextColor(getResources().getColor(R.color.play_mode_create_disable_text_color));
        this.mLayoutPreview.setOnClickListener(null);
        this.mLayoutComplete.setOnClickListener(null);
    }
}
