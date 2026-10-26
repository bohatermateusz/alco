package com.hismith.hismithapp.fragment;

import android.content.Context;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.text.TextUtils;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;
import com.bumptech.glide.Glide;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.bluetooth.BleManager;
import com.hismith.hismithapp.bluetooth.BleProduct;
import com.hismith.hismithapp.product.ProductUtil;
import com.hismith.hismithapp.view.CircleImageView;

/* loaded from: classes3.dex */
public class DeviceUIDoubleFragment extends LazyLoadFragment {
    private ImageView img_device_fun1;
    private ImageView img_device_fun2;
    private CircleImageView img_device_logo;
    private TextView iplay_txt_speed;
    private TextView iplay_txt_vibrate;
    private String mAddress;
    private Context mContext;
    private final int WHAT_SPEED = 1;
    private final int WHAT_VIBRATE = 7;
    private Handler mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.fragment.DeviceUIDoubleFragment.1
        @Override // android.os.Handler.Callback
        public boolean handleMessage(Message message) {
            int i = message.what;
            if (i == 1) {
                DeviceUIDoubleFragment.this.iplay_txt_speed.setText(((Integer) message.obj).intValue() + "");
                return false;
            }
            if (i != 7) {
                return false;
            }
            DeviceUIDoubleFragment.this.iplay_txt_vibrate.setText(((Integer) message.obj).intValue() + "");
            return false;
        }
    });

    public DeviceUIDoubleFragment(String str) {
        this.mAddress = str;
    }

    private void init() {
        BleProduct connBleProduct = BleManager.getInstance(this.mContext).getConnBleProduct(this.mAddress);
        if (connBleProduct == null || connBleProduct.getmProductModel() == null) {
            return;
        }
        Glide.with(this.mContext).load(connBleProduct.getmProductModel().getLogo()).placeholder(R.mipmap.ic_personal_head).into(this.img_device_logo);
        if (ProductUtil.hasMode(connBleProduct.getmProductModel(), 0)) {
            Glide.with(this.mContext).load(connBleProduct.getmProductModel().getFuns()[0].getfIcon()).placeholder(R.mipmap.ic_personal_head).into(this.img_device_fun1);
        }
        if (ProductUtil.hasMode(connBleProduct.getmProductModel(), 1)) {
            Glide.with(this.mContext).load(connBleProduct.getmProductModel().getFuns()[1].getfIcon()).placeholder(R.mipmap.ic_personal_head).into(this.img_device_fun2);
        }
    }

    private void initBleCCData(String str, String str2) {
        int i;
        str.hashCode();
        i = 0;
        switch (str) {
            case "02":
                try {
                    Message message = new Message();
                    message.what = 1;
                    message.obj = 0;
                    this.mHandler.sendMessage(message);
                    Message message2 = new Message();
                    message2.what = 7;
                    message2.obj = 0;
                    this.mHandler.sendMessage(message2);
                    break;
                } catch (Exception e) {
                    e.printStackTrace();
                }
            case "03":
                try {
                    int intValue = Integer.valueOf(str2, 16).intValue();
                    if (intValue > 100) {
                        i = 100;
                    } else if (intValue >= 0) {
                        i = intValue;
                    }
                    Message message3 = new Message();
                    message3.what = 1;
                    message3.obj = Integer.valueOf(i);
                    this.mHandler.sendMessage(message3);
                    break;
                } catch (Exception e2) {
                    e2.printStackTrace();
                    return;
                }
            case "05":
                try {
                    int intValue2 = Integer.valueOf(str2, 16).intValue();
                    if (intValue2 > 100) {
                        i = 100;
                    } else if (intValue2 >= 0) {
                        i = intValue2;
                    }
                    Message message4 = new Message();
                    message4.what = 7;
                    message4.obj = Integer.valueOf(i);
                    this.mHandler.sendMessage(message4);
                    break;
                } catch (Exception e3) {
                    e3.printStackTrace();
                    return;
                }
        }
    }

    private void initView(View view) {
        this.img_device_logo = (CircleImageView) view.findViewById(R.id.img_device_logo);
        this.img_device_fun1 = (ImageView) view.findViewById(R.id.img_device_fun1);
        this.img_device_fun2 = (ImageView) view.findViewById(R.id.img_device_fun2);
        this.iplay_txt_speed = (TextView) view.findViewById(R.id.iplay_txt_speed);
        this.iplay_txt_vibrate = (TextView) view.findViewById(R.id.iplay_txt_vibrate);
    }

    private void setListener() {
    }

    public void buildBleReceData(String str, String str2, String str3, String str4) {
        if (TextUtils.isEmpty(this.mAddress) || !this.mAddress.equals(str)) {
            return;
        }
        if ("CA".equals(str2) || "CC".equals(str2)) {
            initBleCCData(str3, str4);
        }
    }

    @Override // com.hismith.hismithapp.fragment.LazyLoadFragment
    protected void lazyLoad() {
        init();
        setListener();
    }

    @Override // com.hismith.hismithapp.fragment.LazyLoadFragment, androidx.fragment.app.Fragment
    public void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        this.mContext = getContext();
    }

    @Override // androidx.fragment.app.Fragment
    public View onCreateView(LayoutInflater layoutInflater, ViewGroup viewGroup, Bundle bundle) {
        View inflate = layoutInflater.inflate(R.layout.fragment_device_ui_double, (ViewGroup) null);
        initView(inflate);
        return inflate;
    }
}
