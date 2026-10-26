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
public class DeviceUISingleFragment extends LazyLoadFragment {
    private ImageView img_device_fun1;
    private CircleImageView img_device_logo;
    private TextView iplay_txt_speed;
    private String mAddress;
    private Context mContext;
    private final int WHAT_SPEED = 1;
    private Handler mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.fragment.DeviceUISingleFragment.1
        @Override // android.os.Handler.Callback
        public boolean handleMessage(Message message) {
            if (message.what != 1) {
                return false;
            }
            DeviceUISingleFragment.this.iplay_txt_speed.setText(((Integer) message.obj).intValue() + "");
            return false;
        }
    });

    public DeviceUISingleFragment(String str) {
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
    }

    private void initBleAAData(String str, String str2) {
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
                    break;
                } catch (Exception e) {
                    e.printStackTrace();
                    return;
                }
            case "03":
            case "04":
                try {
                    int intValue = Integer.valueOf(str2, 16).intValue();
                    if (intValue > 100) {
                        i = 100;
                    } else if (intValue >= 0) {
                        i = intValue;
                    }
                    Message message2 = new Message();
                    message2.what = 1;
                    message2.obj = Integer.valueOf(i);
                    this.mHandler.sendMessage(message2);
                    break;
                } catch (Exception e2) {
                    e2.printStackTrace();
                    return;
                }
        }
    }

    private void initBleCCData(String str, String str2) {
        str.hashCode();
        int i = 0;
        if (str.equals("02")) {
            try {
                Message message = new Message();
                message.what = 1;
                message.obj = 0;
                this.mHandler.sendMessage(message);
                return;
            } catch (Exception e) {
                e.printStackTrace();
                return;
            }
        }
        if (str.equals("03")) {
            try {
                int intValue = Integer.valueOf(str2, 16).intValue();
                if (intValue > 100) {
                    i = 100;
                } else if (intValue >= 0) {
                    i = intValue;
                }
                Message message2 = new Message();
                message2.what = 1;
                message2.obj = Integer.valueOf(i);
                this.mHandler.sendMessage(message2);
            } catch (Exception e2) {
                e2.printStackTrace();
            }
        }
    }

    private void initView(View view) {
        this.img_device_logo = (CircleImageView) view.findViewById(R.id.img_device_logo);
        this.img_device_fun1 = (ImageView) view.findViewById(R.id.img_device_fun1);
        this.iplay_txt_speed = (TextView) view.findViewById(R.id.iplay_txt_speed);
    }

    private void setListener() {
    }

    public void buildBleReceData(String str, String str2, String str3, String str4) {
        if (TextUtils.isEmpty(this.mAddress) || !this.mAddress.equals(str)) {
            return;
        }
        if ("AA".equals(str2)) {
            initBleAAData(str3, str4);
        } else if ("CA".equals(str2) || "CC".equals(str2)) {
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
        View inflate = layoutInflater.inflate(R.layout.fragment_device_ui_single, (ViewGroup) null);
        initView(inflate);
        return inflate;
    }
}
