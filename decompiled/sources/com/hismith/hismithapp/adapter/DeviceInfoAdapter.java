package com.hismith.hismithapp.adapter;

import android.content.Context;
import android.os.Handler;
import android.os.Message;
import android.text.TextUtils;
import android.view.LayoutInflater;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.recyclerview.widget.RecyclerView;
import com.bumptech.glide.Glide;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.bluetooth.BleManager;
import com.hismith.hismithapp.bluetooth.BleProduct;
import com.hismith.hismithapp.device_rule.SpeedChannel;
import com.hismith.hismithapp.dialog.SelectMotorDialog;
import com.hismith.hismithapp.manager.ControlRuleManager;
import com.hismith.hismithapp.model.BleDeviceModel;
import com.hismith.hismithapp.model.ProductModel;
import com.hismith.hismithapp.product.ProductUtil;
import java.util.ArrayList;
import java.util.List;

/* loaded from: classes3.dex */
public class DeviceInfoAdapter extends RecyclerView.Adapter<ViewHolder> {
    private Context mContext;
    private List<BleDeviceModel> mList;
    private final int WHAT_WRITE_PIU = 1;
    private Handler mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.adapter.DeviceInfoAdapter.1
        @Override // android.os.Handler.Callback
        public boolean handleMessage(Message message) {
            if (message.what != 1) {
                return false;
            }
            String obj = message.obj.toString();
            BleProduct connBleProduct = BleManager.getInstance(DeviceInfoAdapter.this.mContext).getConnBleProduct(obj);
            if (connBleProduct != null) {
                connBleProduct.writeOpenPiu();
            }
            Message message2 = new Message();
            message2.what = 1;
            message2.obj = obj;
            DeviceInfoAdapter.this.mHandler.sendMessageDelayed(message2, 100L);
            return false;
        }
    });

    private class PiuOnTouchListener implements View.OnTouchListener {
        private String address;

        public PiuOnTouchListener(String str) {
            this.address = str;
        }

        @Override // android.view.View.OnTouchListener
        public boolean onTouch(View view, MotionEvent motionEvent) {
            int action = motionEvent.getAction();
            if (action == 0) {
                Message message = new Message();
                message.what = 1;
                message.obj = this.address;
                DeviceInfoAdapter.this.mHandler.sendMessage(message);
            } else if (action == 1) {
                DeviceInfoAdapter.this.mHandler.removeMessages(1);
            }
            return true;
        }
    }

    class ViewHolder extends RecyclerView.ViewHolder {
        private ImageView img_device_fun1;
        private ImageView img_device_fun2;
        private ImageView img_device_logo;
        private ImageView iplay_img_piu;
        private TextView iplay_txt_speed;
        private TextView iplay_txt_vibrate;
        private ImageView mImgBattery;
        private ConstraintLayout mLayoutBattery;
        private View mViewBattery;

        public ViewHolder(View view) {
            super(view);
            this.img_device_logo = (ImageView) view.findViewById(R.id.img_device_logo);
            this.img_device_fun1 = (ImageView) view.findViewById(R.id.img_device_fun1);
            this.img_device_fun2 = (ImageView) view.findViewById(R.id.img_device_fun2);
            this.iplay_txt_speed = (TextView) view.findViewById(R.id.iplay_txt_speed);
            this.iplay_txt_vibrate = (TextView) view.findViewById(R.id.iplay_txt_vibrate);
            this.iplay_img_piu = (ImageView) view.findViewById(R.id.iplay_img_piu);
            this.mLayoutBattery = (ConstraintLayout) view.findViewById(R.id.layout_battery);
            this.mViewBattery = view.findViewById(R.id.view_battery);
            this.mImgBattery = (ImageView) view.findViewById(R.id.img_battery);
        }
    }

    public DeviceInfoAdapter(Context context, List<BleDeviceModel> list) {
        this.mContext = context;
        if (list != null) {
            this.mList = list;
        } else {
            this.mList = new ArrayList();
        }
    }

    private void showSelectMotorDialog(int i, String str, SpeedChannel speedChannel) {
        try {
            new SelectMotorDialog(this.mContext, i, str, ControlRuleManager.getInstance().getDeviceRule(str).m606clone(), speedChannel, "functionSet", null, null).show();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public int getItemCount() {
        return this.mList.size();
    }

    /* renamed from: lambda$onBindViewHolder$0$com-hismith-hismithapp-adapter-DeviceInfoAdapter, reason: not valid java name */
    /* synthetic */ void m578x19298b84(int i, BleDeviceModel bleDeviceModel, View view) {
        showSelectMotorDialog(i, bleDeviceModel.getDeviceAddress(), SpeedChannel.CHANNEL_A);
    }

    /* renamed from: lambda$onBindViewHolder$1$com-hismith-hismithapp-adapter-DeviceInfoAdapter, reason: not valid java name */
    /* synthetic */ void m579xad331a3(int i, BleDeviceModel bleDeviceModel, View view) {
        showSelectMotorDialog(i, bleDeviceModel.getDeviceAddress(), SpeedChannel.CHANNEL_B);
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public /* bridge */ /* synthetic */ void onBindViewHolder(ViewHolder viewHolder, int i, List list) {
        onBindViewHolder2(viewHolder, i, (List<Object>) list);
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public void onBindViewHolder(final ViewHolder viewHolder, final int i) {
        viewHolder.iplay_img_piu.setOnTouchListener(null);
        final BleDeviceModel bleDeviceModel = this.mList.get(i);
        viewHolder.iplay_txt_speed.setText(bleDeviceModel.getSpeed() + "");
        viewHolder.iplay_txt_vibrate.setText(bleDeviceModel.getVibration() + "");
        if (bleDeviceModel.getProductModel() != null) {
            ProductModel productModel = bleDeviceModel.getProductModel();
            Glide.with(this.mContext).load(productModel.getLogo()).placeholder(R.mipmap.ic_app_logo).into(viewHolder.img_device_logo);
            if (ProductUtil.hasMode(productModel, 0)) {
                if (!TextUtils.isEmpty(productModel.getFuns()[0].getfIcon())) {
                    Glide.with(this.mContext).load(productModel.getFuns()[0].getfIcon()).circleCrop().placeholder(R.drawable.bg_motor_circle_black).into(viewHolder.img_device_fun1);
                } else if (productModel.getFuns()[0].getfIconLocation() > 0) {
                    Glide.with(this.mContext).load(Integer.valueOf(productModel.getFuns()[0].getfIconLocation())).circleCrop().placeholder(R.drawable.bg_motor_circle_black).into(viewHolder.img_device_fun1);
                }
            }
            if (ProductUtil.isFType(this.mContext, bleDeviceModel.getDeviceAddress())) {
                viewHolder.iplay_txt_speed.setVisibility(8);
                viewHolder.iplay_img_piu.setVisibility(0);
            } else {
                viewHolder.iplay_txt_speed.setVisibility(0);
                viewHolder.iplay_img_piu.setVisibility(8);
            }
            if (ProductUtil.hasMode(productModel, 1)) {
                if (!TextUtils.isEmpty(productModel.getFuns()[1].getfIcon())) {
                    Glide.with(this.mContext).load(productModel.getFuns()[1].getfIcon()).circleCrop().placeholder(R.drawable.bg_motor_circle_black).into(viewHolder.img_device_fun2);
                } else if (productModel.getFuns()[1].getfIconLocation() > 0) {
                    Glide.with(this.mContext).load(Integer.valueOf(productModel.getFuns()[1].getfIconLocation())).circleCrop().placeholder(R.drawable.bg_motor_circle_black).into(viewHolder.img_device_fun2);
                }
                viewHolder.img_device_fun2.setVisibility(0);
                viewHolder.iplay_txt_vibrate.setVisibility(0);
            } else {
                viewHolder.img_device_fun2.setVisibility(8);
                viewHolder.iplay_txt_vibrate.setVisibility(8);
            }
            viewHolder.img_device_fun1.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.DeviceInfoAdapter$$ExternalSyntheticLambda0
                @Override // android.view.View.OnClickListener
                public final void onClick(View view) {
                    DeviceInfoAdapter.this.m578x19298b84(i, bleDeviceModel, view);
                }
            });
            viewHolder.img_device_fun2.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.DeviceInfoAdapter$$ExternalSyntheticLambda1
                @Override // android.view.View.OnClickListener
                public final void onClick(View view) {
                    DeviceInfoAdapter.this.m579xad331a3(i, bleDeviceModel, view);
                }
            });
        }
        if (this.mList.size() <= i || this.mList.get(i).getBattery() < 0) {
            viewHolder.mLayoutBattery.setVisibility(8);
        } else {
            viewHolder.mLayoutBattery.setVisibility(0);
            viewHolder.mViewBattery.post(new Runnable() { // from class: com.hismith.hismithapp.adapter.DeviceInfoAdapter.3
                @Override // java.lang.Runnable
                public void run() {
                    int battery = DeviceInfoAdapter.this.mList.size() > i ? ((BleDeviceModel) DeviceInfoAdapter.this.mList.get(i)).getBattery() : 0;
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
        }
        viewHolder.iplay_img_piu.setOnTouchListener(new PiuOnTouchListener(this.mList.get(i).getDeviceAddress()));
    }

    /* renamed from: onBindViewHolder, reason: avoid collision after fix types in other method */
    public void onBindViewHolder2(final ViewHolder viewHolder, final int i, List<Object> list) {
        if (list.isEmpty()) {
            onBindViewHolder(viewHolder, i);
            return;
        }
        for (Object obj : list) {
            if (obj instanceof String) {
                String str = (String) obj;
                if ("value".equals(str)) {
                    viewHolder.iplay_txt_speed.setText(this.mList.get(i).getSpeed() + "");
                    viewHolder.iplay_txt_vibrate.setText(this.mList.get(i).getVibration() + "");
                } else if (!"mode".equals(str)) {
                    if ("piu".equals(str)) {
                        if (this.mList.get(i).getPiu() == 1) {
                            viewHolder.iplay_img_piu.setImageResource(R.mipmap.ic_iplay_piu_select);
                        } else {
                            viewHolder.iplay_img_piu.setImageResource(R.mipmap.ic_iplay_piu);
                        }
                    } else if ("battery".equals(str)) {
                        if (this.mList.size() <= i || this.mList.get(i).getBattery() < 0) {
                            viewHolder.mLayoutBattery.setVisibility(8);
                        } else {
                            viewHolder.mLayoutBattery.setVisibility(0);
                            viewHolder.mViewBattery.post(new Runnable() { // from class: com.hismith.hismithapp.adapter.DeviceInfoAdapter.2
                                @Override // java.lang.Runnable
                                public void run() {
                                    int battery = DeviceInfoAdapter.this.mList.size() > i ? ((BleDeviceModel) DeviceInfoAdapter.this.mList.get(i)).getBattery() : 0;
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
                        }
                    }
                }
            }
        }
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public ViewHolder onCreateViewHolder(ViewGroup viewGroup, int i) {
        return new ViewHolder(LayoutInflater.from(viewGroup.getContext()).inflate(R.layout.item_device_info, viewGroup, false));
    }
}
