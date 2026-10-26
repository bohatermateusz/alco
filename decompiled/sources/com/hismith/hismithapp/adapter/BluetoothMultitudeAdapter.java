package com.hismith.hismithapp.adapter;

import android.bluetooth.BluetoothDevice;
import android.content.Context;
import android.text.TextUtils;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.CompoundButton;
import android.widget.ImageView;
import android.widget.RadioButton;
import android.widget.TextView;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.recyclerview.widget.RecyclerView;
import com.bumptech.glide.Glide;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.bluetooth.BleManager;
import com.hismith.hismithapp.bluetooth.BleProduct;
import com.hismith.hismithapp.device_rule.SpeedChannel;
import com.hismith.hismithapp.manager.ControlRuleManager;
import com.hismith.hismithapp.model.BleDeviceModel;
import com.hismith.hismithapp.product.ProductUtil;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.ToastUtil;
import com.hismith.hismithapp.view.CenterSeekBar;
import java.util.List;

/* loaded from: classes3.dex */
public class BluetoothMultitudeAdapter extends RecyclerView.Adapter<ViewHolder> {
    private boolean isClicked = false;
    private int leaderPosition = -1;
    private Context mContext;
    private List<BleDeviceModel> mList;
    private MotorClickListener motorClickListener;
    private OnEditNameListener onEditNameListener;

    public interface MotorClickListener {
        void onMotor1Click(int i, String str);

        void onMotor2Click(int i, String str);
    }

    public interface OnEditNameListener {
        void onEditName(BluetoothDevice bluetoothDevice, String str, int i);
    }

    class ViewHolder extends RecyclerView.ViewHolder {
        private Button btn_device_conn;
        private ImageView img_device;
        private ImageView img_motor1;
        private ImageView img_motor1_link;
        private ImageView img_motor1_motor1;
        private ImageView img_motor2;
        private ImageView img_motor2_link;
        private ImageView img_motor2_motor2;
        private TextView item_txt_rename;
        private ImageView mBgBattery;
        private ImageView mImgBattery;
        private ConstraintLayout mLayoutBattery;
        private View mViewBattery;
        private RadioButton radio_main;
        private CenterSeekBar seek_motor1_scale;
        private CenterSeekBar seek_motor2_scale;
        private TextView txt_device_name;
        private TextView txt_motor1_scale;
        private TextView txt_motor2_scale;

        public ViewHolder(View view) {
            super(view);
            this.txt_device_name = (TextView) view.findViewById(R.id.txt_device_name);
            this.radio_main = (RadioButton) view.findViewById(R.id.radio_main);
            this.btn_device_conn = (Button) view.findViewById(R.id.btn_device_conn);
            this.img_device = (ImageView) view.findViewById(R.id.img_device);
            this.img_motor1 = (ImageView) view.findViewById(R.id.img_motor1);
            this.img_motor1_link = (ImageView) view.findViewById(R.id.img_motor1_link);
            this.img_motor1_motor1 = (ImageView) view.findViewById(R.id.img_motor1_motor1);
            this.img_motor2 = (ImageView) view.findViewById(R.id.img_motor2);
            this.img_motor2_link = (ImageView) view.findViewById(R.id.img_motor2_link);
            this.img_motor2_motor2 = (ImageView) view.findViewById(R.id.img_motor2_motor2);
            this.seek_motor1_scale = (CenterSeekBar) view.findViewById(R.id.seek_motor1_scale);
            this.seek_motor2_scale = (CenterSeekBar) view.findViewById(R.id.seek_motor2_scale);
            this.txt_motor1_scale = (TextView) view.findViewById(R.id.txt_motor1_scale);
            this.txt_motor2_scale = (TextView) view.findViewById(R.id.txt_motor2_scale);
            this.mLayoutBattery = (ConstraintLayout) view.findViewById(R.id.layout_battery);
            this.mViewBattery = view.findViewById(R.id.view_battery);
            this.mImgBattery = (ImageView) view.findViewById(R.id.img_battery);
            this.mBgBattery = (ImageView) view.findViewById(R.id.bg_battery);
            this.item_txt_rename = (TextView) view.findViewById(R.id.item_txt_rename);
            LanguageUtil.setText(BluetoothMultitudeAdapter.this.mContext, this.radio_main, "master");
            LanguageUtil.setText(BluetoothMultitudeAdapter.this.mContext, this.item_txt_rename, "edit");
        }

        public void bind(int i, BleDeviceModel bleDeviceModel, ControlRuleManager.ControlRule controlRule) {
            this.img_device.setImageResource(R.mipmap.ic_app_logo);
            this.txt_device_name.setText(bleDeviceModel.getDeviceName());
            this.txt_device_name.requestLayout();
            if (bleDeviceModel.getProductModel() == null) {
                LogUtil.e("onBindViewHolder bind productModel :" + bleDeviceModel.getProductModel());
            }
            if (bleDeviceModel.getProductModel() != null) {
                Glide.with(BluetoothMultitudeAdapter.this.mContext).load(bleDeviceModel.getProductModel().getLogo()).placeholder(R.mipmap.ic_app_logo).into(this.img_device);
            }
            int isConnectedBleProduct = BleManager.getInstance(BluetoothMultitudeAdapter.this.mContext).isConnectedBleProduct(bleDeviceModel.getDeviceAddress());
            RadioButton radioButton = this.radio_main;
            if (isConnectedBleProduct != 2) {
                radioButton.setVisibility(8);
                this.img_motor1.setVisibility(8);
                this.img_motor1_link.setVisibility(8);
                this.img_motor1_motor1.setVisibility(8);
                this.img_motor2.setVisibility(8);
                this.img_motor2_link.setVisibility(8);
                this.img_motor2_motor2.setVisibility(8);
                this.mLayoutBattery.setVisibility(8);
                return;
            }
            radioButton.setVisibility(0);
            if (ControlRuleManager.getInstance().isLeader(bleDeviceModel.getDeviceAddress())) {
                BluetoothMultitudeAdapter.this.leaderPosition = i;
                this.radio_main.setChecked(true);
                boolean hasFun = ProductUtil.hasFun(0);
                ImageView imageView = this.img_motor1;
                if (hasFun) {
                    imageView.setVisibility(0);
                    if (!TextUtils.isEmpty(ProductUtil.getProductFun(0).getfIcon())) {
                        Glide.with(BluetoothMultitudeAdapter.this.mContext).load(ProductUtil.getProductFun(0).getfIcon()).circleCrop().placeholder(R.drawable.bg_motor_circle_black).into(this.img_motor1);
                    } else if (ProductUtil.getProductFun(0).getfIconLocation() > 0) {
                        Glide.with(BluetoothMultitudeAdapter.this.mContext).load(Integer.valueOf(ProductUtil.getProductFun(0).getfIconLocation())).circleCrop().placeholder(R.drawable.bg_motor_circle_black).into(this.img_motor1);
                    }
                } else {
                    imageView.setVisibility(8);
                }
                this.img_motor1_link.setVisibility(8);
                this.img_motor1_motor1.setVisibility(8);
                boolean hasFun2 = ProductUtil.hasFun(1);
                ImageView imageView2 = this.img_motor2;
                if (hasFun2) {
                    imageView2.setVisibility(0);
                    if (!TextUtils.isEmpty(ProductUtil.getProductFun(1).getfIcon())) {
                        Glide.with(BluetoothMultitudeAdapter.this.mContext).load(ProductUtil.getProductFun(1).getfIcon()).circleCrop().placeholder(R.drawable.bg_motor_circle_black).into(this.img_motor2);
                    } else if (ProductUtil.getProductFun(1).getfIconLocation() > 0) {
                        Glide.with(BluetoothMultitudeAdapter.this.mContext).load(Integer.valueOf(ProductUtil.getProductFun(1).getfIconLocation())).circleCrop().placeholder(R.drawable.bg_motor_circle_black).into(this.img_motor2);
                    }
                } else {
                    imageView2.setVisibility(8);
                }
                this.img_motor2_link.setVisibility(8);
                this.img_motor2_motor2.setVisibility(8);
            } else {
                this.radio_main.setChecked(false);
                boolean hasFun3 = ProductUtil.hasFun(bleDeviceModel.getProductModel(), 0);
                ImageView imageView3 = this.img_motor1;
                if (hasFun3) {
                    imageView3.setVisibility(0);
                    this.img_motor1_link.setVisibility(0);
                    this.img_motor1_motor1.setVisibility(0);
                    if (!TextUtils.isEmpty(ProductUtil.getProductFun(bleDeviceModel.getProductModel(), 0).getfIcon())) {
                        Glide.with(BluetoothMultitudeAdapter.this.mContext).load(ProductUtil.getProductFun(bleDeviceModel.getProductModel(), 0).getfIcon()).circleCrop().placeholder(R.drawable.bg_motor_circle_black).into(this.img_motor1);
                    } else if (ProductUtil.getProductFun(bleDeviceModel.getProductModel(), 0).getfIconLocation() > 0) {
                        Glide.with(BluetoothMultitudeAdapter.this.mContext).load(Integer.valueOf(ProductUtil.getProductFun(bleDeviceModel.getProductModel(), 0).getfIconLocation())).circleCrop().placeholder(R.drawable.bg_motor_circle_black).into(this.img_motor1);
                    }
                    if (controlRule.getSpeedChannel() == SpeedChannel.CHANNEL_A) {
                        if (ProductUtil.hasFun(0)) {
                            if (!TextUtils.isEmpty(ProductUtil.getProductFun(0).getfIcon())) {
                                Glide.with(BluetoothMultitudeAdapter.this.mContext).load(ProductUtil.getProductFun(0).getfIcon()).circleCrop().placeholder(R.drawable.bg_motor_circle_black).into(this.img_motor1_motor1);
                            } else if (ProductUtil.getProductFun(0).getfIconLocation() > 0) {
                                Glide.with(BluetoothMultitudeAdapter.this.mContext).load(Integer.valueOf(ProductUtil.getProductFun(0).getfIconLocation())).circleCrop().placeholder(R.drawable.bg_motor_circle_black).into(this.img_motor1_motor1);
                            }
                        }
                    } else if (controlRule.getSpeedChannel() == SpeedChannel.CHANNEL_B) {
                        if (ProductUtil.hasFun(1)) {
                            if (!TextUtils.isEmpty(ProductUtil.getProductFun(1).getfIcon())) {
                                Glide.with(BluetoothMultitudeAdapter.this.mContext).load(ProductUtil.getProductFun(1).getfIcon()).circleCrop().placeholder(R.drawable.bg_motor_circle_black).into(this.img_motor1_motor1);
                            } else if (ProductUtil.getProductFun(1).getfIconLocation() > 0) {
                                Glide.with(BluetoothMultitudeAdapter.this.mContext).load(Integer.valueOf(ProductUtil.getProductFun(1).getfIconLocation())).circleCrop().placeholder(R.drawable.bg_motor_circle_black).into(this.img_motor1_motor1);
                            }
                        }
                    } else if (controlRule.getSpeedChannel() == SpeedChannel.CHANNEL_NONE) {
                        this.img_motor1_motor1.setImageResource(R.mipmap.ic_iplay_close_mode);
                    }
                } else {
                    imageView3.setVisibility(8);
                    this.img_motor1_link.setVisibility(8);
                    this.img_motor1_motor1.setVisibility(8);
                }
                boolean hasFun4 = ProductUtil.hasFun(bleDeviceModel.getProductModel(), 1);
                ImageView imageView4 = this.img_motor2;
                if (hasFun4) {
                    imageView4.setVisibility(0);
                    this.img_motor2_link.setVisibility(0);
                    this.img_motor2_motor2.setVisibility(0);
                    if (!TextUtils.isEmpty(ProductUtil.getProductFun(bleDeviceModel.getProductModel(), 1).getfIcon())) {
                        Glide.with(BluetoothMultitudeAdapter.this.mContext).load(ProductUtil.getProductFun(bleDeviceModel.getProductModel(), 1).getfIcon()).circleCrop().placeholder(R.drawable.bg_motor_circle_black).into(this.img_motor2);
                    } else if (ProductUtil.getProductFun(bleDeviceModel.getProductModel(), 1).getfIconLocation() > 0) {
                        Glide.with(BluetoothMultitudeAdapter.this.mContext).load(Integer.valueOf(ProductUtil.getProductFun(bleDeviceModel.getProductModel(), 1).getfIconLocation())).circleCrop().placeholder(R.drawable.bg_motor_circle_black).into(this.img_motor2);
                    }
                    if (controlRule.getVibrateChannel() == SpeedChannel.CHANNEL_A) {
                        if (ProductUtil.hasFun(0)) {
                            if (!TextUtils.isEmpty(ProductUtil.getProductFun(0).getfIcon())) {
                                Glide.with(BluetoothMultitudeAdapter.this.mContext).load(ProductUtil.getProductFun(0).getfIcon()).circleCrop().placeholder(R.drawable.bg_motor_circle_black).into(this.img_motor2_motor2);
                            } else if (ProductUtil.getProductFun(0).getfIconLocation() > 0) {
                                Glide.with(BluetoothMultitudeAdapter.this.mContext).load(Integer.valueOf(ProductUtil.getProductFun(0).getfIconLocation())).circleCrop().placeholder(R.drawable.bg_motor_circle_black).into(this.img_motor2_motor2);
                            }
                        }
                    } else if (controlRule.getVibrateChannel() == SpeedChannel.CHANNEL_B) {
                        if (ProductUtil.hasFun(1)) {
                            if (!TextUtils.isEmpty(ProductUtil.getProductFun(1).getfIcon())) {
                                Glide.with(BluetoothMultitudeAdapter.this.mContext).load(ProductUtil.getProductFun(1).getfIcon()).circleCrop().placeholder(R.drawable.bg_motor_circle_black).into(this.img_motor2_motor2);
                            } else if (ProductUtil.getProductFun(1).getfIconLocation() > 0) {
                                Glide.with(BluetoothMultitudeAdapter.this.mContext).load(Integer.valueOf(ProductUtil.getProductFun(1).getfIconLocation())).circleCrop().placeholder(R.drawable.bg_motor_circle_black).into(this.img_motor2_motor2);
                            }
                        }
                    } else if (controlRule.getSpeedChannel() == SpeedChannel.CHANNEL_NONE) {
                        this.img_motor2_motor2.setImageResource(R.mipmap.ic_iplay_close_mode);
                    }
                } else {
                    imageView4.setVisibility(8);
                    this.img_motor2_link.setVisibility(8);
                    this.img_motor2_motor2.setVisibility(8);
                }
            }
            BleProduct connBleProduct = BleManager.getInstance(BluetoothMultitudeAdapter.this.mContext).getConnBleProduct(((BleDeviceModel) BluetoothMultitudeAdapter.this.mList.get(i)).getDeviceAddress());
            if (connBleProduct == null) {
                this.mLayoutBattery.setVisibility(8);
                return;
            }
            final int battery = connBleProduct.getConnBleDevice().getBattery();
            ConstraintLayout constraintLayout = this.mLayoutBattery;
            if (battery < 0) {
                constraintLayout.setVisibility(8);
            } else {
                constraintLayout.setVisibility(0);
                this.mViewBattery.post(new Runnable() { // from class: com.hismith.hismithapp.adapter.BluetoothMultitudeAdapter.ViewHolder.1
                    @Override // java.lang.Runnable
                    public void run() {
                        ViewGroup.LayoutParams layoutParams = ViewHolder.this.mImgBattery.getLayoutParams();
                        int width = (int) (ViewHolder.this.mViewBattery.getWidth() * (battery / 10.0d));
                        ViewHolder viewHolder = ViewHolder.this;
                        if (width >= 0) {
                            viewHolder.mImgBattery.setVisibility(0);
                        } else {
                            viewHolder.mImgBattery.setVisibility(8);
                        }
                        layoutParams.width = width;
                        ViewHolder.this.mImgBattery.setLayoutParams(layoutParams);
                    }
                });
            }
        }
    }

    public BluetoothMultitudeAdapter(Context context, List<BleDeviceModel> list) {
        this.mContext = context;
        this.mList = list;
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public int getItemCount() {
        List<BleDeviceModel> list = this.mList;
        if (list != null) {
            return list.size();
        }
        return 0;
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public /* bridge */ /* synthetic */ void onBindViewHolder(ViewHolder viewHolder, int i, List list) {
        onBindViewHolder2(viewHolder, i, (List<Object>) list);
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public void onBindViewHolder(ViewHolder viewHolder, final int i) {
        viewHolder.radio_main.setOnCheckedChangeListener(null);
        final BleDeviceModel bleDeviceModel = this.mList.get(i);
        viewHolder.bind(i, bleDeviceModel, ControlRuleManager.getInstance().getDeviceRule(bleDeviceModel.getDeviceAddress()));
        LogUtil.e("onBindViewHolder position:" + i + ", name:" + bleDeviceModel.getDeviceName() + ", status:" + BleManager.getInstance(this.mContext).isConnectedBleProduct(bleDeviceModel.getDeviceAddress()));
        int isConnectedBleProduct = BleManager.getInstance(this.mContext).isConnectedBleProduct(bleDeviceModel.getDeviceAddress());
        Context context = this.mContext;
        if (isConnectedBleProduct == 2) {
            LanguageUtil.setText(context, viewHolder.btn_device_conn, "connected");
            viewHolder.btn_device_conn.setTextColor(this.mContext.getResources().getColor(R.color.app_style_color));
            viewHolder.btn_device_conn.setBackgroundResource(R.drawable.bg_corners_border_style_15);
            viewHolder.btn_device_conn.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.BluetoothMultitudeAdapter.2
                @Override // android.view.View.OnClickListener
                public void onClick(View view) {
                    BleManager.getInstance(BluetoothMultitudeAdapter.this.mContext).disconnectGatt(bleDeviceModel.getBluetoothDevice(), bleDeviceModel.getDeviceName());
                }
            });
        } else {
            int isConnectedBleProduct2 = BleManager.getInstance(context).isConnectedBleProduct(bleDeviceModel.getDeviceAddress());
            Context context2 = this.mContext;
            if (isConnectedBleProduct2 == 0) {
                LanguageUtil.setText(context2, viewHolder.btn_device_conn, "connection");
                viewHolder.btn_device_conn.setTextColor(this.mContext.getResources().getColor(R.color.white));
                viewHolder.btn_device_conn.setBackgroundResource(R.drawable.bg_edit_style_15);
                viewHolder.btn_device_conn.setAlpha(1.0f);
                viewHolder.btn_device_conn.setEnabled(true);
                viewHolder.btn_device_conn.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.BluetoothMultitudeAdapter.3
                    @Override // android.view.View.OnClickListener
                    public void onClick(View view) {
                        boolean isExceedMaxConnection = BleManager.getInstance(BluetoothMultitudeAdapter.this.mContext).isExceedMaxConnection();
                        BluetoothMultitudeAdapter bluetoothMultitudeAdapter = BluetoothMultitudeAdapter.this;
                        if (isExceedMaxConnection) {
                            ToastUtil.showShortToast(BluetoothMultitudeAdapter.this.mContext, LanguageUtil.getLanguageByKey(bluetoothMultitudeAdapter.mContext, "maxConnected"));
                        } else {
                            if (bluetoothMultitudeAdapter.isClicked) {
                                return;
                            }
                            BluetoothMultitudeAdapter.this.isClicked = true;
                            BluetoothMultitudeAdapter.this.notifyItemChanged(i);
                            BleManager.getInstance(BluetoothMultitudeAdapter.this.mContext).connectGatt(bleDeviceModel.getBluetoothDevice(), bleDeviceModel.getDeviceName(), bleDeviceModel.getDeviceCode());
                        }
                    }
                });
            } else if (BleManager.getInstance(context2).isConnectedBleProduct(bleDeviceModel.getDeviceAddress()) == 1) {
                viewHolder.btn_device_conn.setTextColor(this.mContext.getResources().getColor(R.color.app_style_color));
                viewHolder.btn_device_conn.setBackgroundResource(R.drawable.bg_corners_border_style_15);
                LanguageUtil.setText(this.mContext, viewHolder.btn_device_conn, "connecting");
            }
        }
        viewHolder.radio_main.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() { // from class: com.hismith.hismithapp.adapter.BluetoothMultitudeAdapter.4
            @Override // android.widget.CompoundButton.OnCheckedChangeListener
            public void onCheckedChanged(CompoundButton compoundButton, boolean z) {
                BleManager.getInstance(BluetoothMultitudeAdapter.this.mContext).selectApplicationProduct(bleDeviceModel.getDeviceAddress());
                BluetoothMultitudeAdapter.this.notifyDataSetChanged();
                BluetoothMultitudeAdapter.this.leaderPosition = i;
            }
        });
        viewHolder.img_motor1_motor1.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.BluetoothMultitudeAdapter.5
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                if (BluetoothMultitudeAdapter.this.motorClickListener != null) {
                    BluetoothMultitudeAdapter.this.motorClickListener.onMotor1Click(i, bleDeviceModel.getDeviceAddress());
                }
            }
        });
        viewHolder.img_motor2_motor2.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.BluetoothMultitudeAdapter.6
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                if (BluetoothMultitudeAdapter.this.motorClickListener != null) {
                    BluetoothMultitudeAdapter.this.motorClickListener.onMotor2Click(i, bleDeviceModel.getDeviceAddress());
                }
            }
        });
        viewHolder.item_txt_rename.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.BluetoothMultitudeAdapter.7
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                if (BluetoothMultitudeAdapter.this.onEditNameListener != null) {
                    BluetoothMultitudeAdapter.this.onEditNameListener.onEditName(bleDeviceModel.getBluetoothDevice(), bleDeviceModel.getDeviceName(), i);
                }
            }
        });
    }

    /* renamed from: onBindViewHolder, reason: avoid collision after fix types in other method */
    public void onBindViewHolder2(final ViewHolder viewHolder, int i, List<Object> list) {
        final int battery;
        if (list.isEmpty()) {
            onBindViewHolder(viewHolder, i);
            return;
        }
        for (Object obj : list) {
            if ((obj instanceof String) && "battery".equals((String) obj)) {
                if (BleManager.getInstance(this.mContext).getConnBleProduct(this.mList.get(i).getDeviceAddress()) != null && (battery = BleManager.getInstance(this.mContext).getConnBleProduct(this.mList.get(i).getDeviceAddress()).getConnBleDevice().getBattery()) > 0) {
                    viewHolder.mLayoutBattery.setVisibility(0);
                    viewHolder.mViewBattery.post(new Runnable() { // from class: com.hismith.hismithapp.adapter.BluetoothMultitudeAdapter.1
                        @Override // java.lang.Runnable
                        public void run() {
                            ViewGroup.LayoutParams layoutParams = viewHolder.mImgBattery.getLayoutParams();
                            int width = (int) (viewHolder.mViewBattery.getWidth() * (battery / 10.0d));
                            ViewHolder viewHolder2 = viewHolder;
                            if (width >= 0) {
                                viewHolder2.mImgBattery.setVisibility(0);
                            } else {
                                viewHolder2.mImgBattery.setVisibility(8);
                            }
                            layoutParams.width = width;
                            viewHolder.mImgBattery.setLayoutParams(layoutParams);
                        }
                    });
                    return;
                }
                viewHolder.mLayoutBattery.setVisibility(8);
            }
        }
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public ViewHolder onCreateViewHolder(ViewGroup viewGroup, int i) {
        return new ViewHolder(LayoutInflater.from(viewGroup.getContext()).inflate(R.layout.item_bluetooth_multitude, viewGroup, false));
    }

    public void setClicked(boolean z) {
        this.isClicked = z;
    }

    public void setMotorClickListener(MotorClickListener motorClickListener) {
        this.motorClickListener = motorClickListener;
    }

    public void setOnEditNameListener(OnEditNameListener onEditNameListener) {
        this.onEditNameListener = onEditNameListener;
    }
}
