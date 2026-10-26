package com.hismith.hismithapp.fragment;

import android.animation.Animator;
import android.animation.AnimatorListenerAdapter;
import android.animation.ValueAnimator;
import android.content.Context;
import android.graphics.Rect;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.TouchDelegate;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.adapter.DeviceInfoAdapter;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.bluetooth.BleManager;
import com.hismith.hismithapp.bluetooth.BleProduct;
import com.hismith.hismithapp.model.BleDeviceModel;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.MeasureUtil;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/* loaded from: classes3.dex */
public class DeviceUIListFragment extends LazyLoadFragment implements View.OnClickListener {
    private ImageView img_drag;
    private ImageView img_small;
    private ConstraintLayout layout_device;
    private ConstraintLayout layout_root;
    private DeviceInfoAdapter mAdapter;
    private Context mContext;
    private LinearLayoutManager mLayoutManager;
    private int mLayoutWidth;
    private List<BleDeviceModel> mList;
    private RecyclerView recycler_device;
    private boolean isSmall = false;
    private boolean isAnimating = false;

    private void animateTranslation(int i, int i2) {
        this.isAnimating = true;
        ValueAnimator ofInt = ValueAnimator.ofInt(i, i2);
        ofInt.setDuration(250L);
        ofInt.addUpdateListener(new ValueAnimator.AnimatorUpdateListener() { // from class: com.hismith.hismithapp.fragment.DeviceUIListFragment$$ExternalSyntheticLambda0
            @Override // android.animation.ValueAnimator.AnimatorUpdateListener
            public final void onAnimationUpdate(ValueAnimator valueAnimator) {
                DeviceUIListFragment.this.m597x2d0c3109(valueAnimator);
            }
        });
        ofInt.addListener(new AnimatorListenerAdapter() { // from class: com.hismith.hismithapp.fragment.DeviceUIListFragment.3
            @Override // android.animation.AnimatorListenerAdapter, android.animation.Animator.AnimatorListener
            public void onAnimationEnd(Animator animator) {
                DeviceUIListFragment.this.isAnimating = false;
            }
        });
        ofInt.start();
    }

    private void init() {
        LinearLayoutManager linearLayoutManager = new LinearLayoutManager(this.mContext);
        this.mLayoutManager = linearLayoutManager;
        this.recycler_device.setLayoutManager(linearLayoutManager);
        this.recycler_device.setNestedScrollingEnabled(false);
        DeviceInfoAdapter deviceInfoAdapter = new DeviceInfoAdapter(this.mContext, this.mList);
        this.mAdapter = deviceInfoAdapter;
        this.recycler_device.setAdapter(deviceInfoAdapter);
        this.layout_device.post(new Runnable() { // from class: com.hismith.hismithapp.fragment.DeviceUIListFragment.1
            @Override // java.lang.Runnable
            public void run() {
                DeviceUIListFragment deviceUIListFragment = DeviceUIListFragment.this;
                deviceUIListFragment.mLayoutWidth = deviceUIListFragment.layout_device.getWidth();
            }
        });
        this.img_drag.post(new Runnable() { // from class: com.hismith.hismithapp.fragment.DeviceUIListFragment.2
            @Override // java.lang.Runnable
            public void run() {
                Rect rect = new Rect();
                DeviceUIListFragment.this.img_drag.getHitRect(rect);
                int dp2px = MeasureUtil.dp2px(DeviceUIListFragment.this.mContext, 130.0f);
                rect.left -= dp2px;
                rect.top -= dp2px;
                rect.right += dp2px;
                rect.bottom += dp2px;
                DeviceUIListFragment.this.layout_device.setTouchDelegate(new TouchDelegate(rect, DeviceUIListFragment.this.img_drag));
            }
        });
    }

    private void initBleAAData(int i, String str, String str2) {
        int intValue;
        intValue = Integer.valueOf(str2, 16).intValue();
        str.hashCode();
        switch (str) {
            case "02":
                this.mList.get(i).setSpeed(0);
                this.mList.get(i).setVibration(0);
                DeviceInfoAdapter deviceInfoAdapter = this.mAdapter;
                if (deviceInfoAdapter != null) {
                    deviceInfoAdapter.notifyItemChanged(i, "value");
                    break;
                }
                break;
            case "03":
            case "04":
                this.mList.get(i).setSpeed(intValue);
                DeviceInfoAdapter deviceInfoAdapter2 = this.mAdapter;
                if (deviceInfoAdapter2 != null) {
                    deviceInfoAdapter2.notifyItemChanged(i, "value");
                    break;
                }
                break;
        }
    }

    private void initBleCCData(int i, String str, String str2, String str3) {
        int intValue;
        LogUtil.e("initBleCCData name:" + this.mList.get(i).getDeviceName() + ", data0:" + str2 + ", data1:" + str3);
        intValue = Integer.valueOf(str3, 16).intValue();
        str2.hashCode();
        switch (str2) {
            case "00":
                if ("CA".equals(str) || "CC".equals(str)) {
                    this.mList.get(i).setBattery(intValue);
                    DeviceInfoAdapter deviceInfoAdapter = this.mAdapter;
                    if (deviceInfoAdapter != null) {
                        deviceInfoAdapter.notifyItemChanged(i, "battery");
                        break;
                    }
                }
                break;
            case "02":
                this.mList.get(i).setSpeed(0);
                this.mList.get(i).setVibration(0);
                DeviceInfoAdapter deviceInfoAdapter2 = this.mAdapter;
                if (deviceInfoAdapter2 != null) {
                    deviceInfoAdapter2.notifyItemChanged(i, "value");
                    break;
                }
                break;
            case "03":
                this.mList.get(i).setSpeed(intValue);
                DeviceInfoAdapter deviceInfoAdapter3 = this.mAdapter;
                if (deviceInfoAdapter3 != null) {
                    deviceInfoAdapter3.notifyItemChanged(i, "value");
                    break;
                }
                break;
            case "05":
                this.mList.get(i).setVibration(intValue);
                DeviceInfoAdapter deviceInfoAdapter4 = this.mAdapter;
                if (deviceInfoAdapter4 != null) {
                    deviceInfoAdapter4.notifyItemChanged(i, "value");
                    break;
                }
                break;
            case "0B":
                this.mList.get(i).setPiu(intValue);
                DeviceInfoAdapter deviceInfoAdapter5 = this.mAdapter;
                if (deviceInfoAdapter5 != null) {
                    deviceInfoAdapter5.notifyItemChanged(i, "piu");
                    break;
                }
                break;
        }
    }

    private void setListener() {
        this.img_small.setOnClickListener(this);
    }

    public void buildBleReceData(String str, String str2, String str3, String str4) {
        int i = 0;
        while (true) {
            List<BleDeviceModel> list = this.mList;
            if (list == null || i >= list.size()) {
                return;
            }
            if (this.mList.get(i).getDeviceAddress().equals(str)) {
                if ("AA".equals(str2)) {
                    initBleAAData(i, str3, str4);
                } else if ("CA".equals(str2) || "CC".equals(str2)) {
                    initBleCCData(i, str2, str3, str4);
                }
            }
            i++;
        }
    }

    public List<BleDeviceModel> getmList() {
        return this.mList;
    }

    /* renamed from: lambda$animateTranslation$0$com-hismith-hismithapp-fragment-DeviceUIListFragment, reason: not valid java name */
    /* synthetic */ void m597x2d0c3109(ValueAnimator valueAnimator) {
        this.layout_root.setTranslationX(((Integer) valueAnimator.getAnimatedValue()).intValue());
    }

    @Override // com.hismith.hismithapp.fragment.LazyLoadFragment
    protected void lazyLoad() {
        init();
        setListener();
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        if (view.getId() != R.id.img_small || this.isAnimating) {
            return;
        }
        if (this.mLayoutWidth == 0) {
            this.mLayoutWidth = this.layout_device.getWidth();
        }
        boolean z = this.isSmall;
        this.isSmall = !z;
        ImageView imageView = this.img_small;
        if (z) {
            imageView.setImageResource(R.mipmap.ic_device_small_left);
            animateTranslation(-this.mLayoutWidth, 0);
        } else {
            imageView.setImageResource(R.mipmap.ic_device_full_left);
            animateTranslation(0, -this.mLayoutWidth);
        }
    }

    @Override // com.hismith.hismithapp.fragment.LazyLoadFragment, androidx.fragment.app.Fragment
    public void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        this.mContext = getContext();
        this.mList = new ArrayList();
        refreshDeviceInfo();
    }

    @Override // androidx.fragment.app.Fragment
    public View onCreateView(LayoutInflater layoutInflater, ViewGroup viewGroup, Bundle bundle) {
        View inflate = layoutInflater.inflate(R.layout.fragment_device_ui_list, (ViewGroup) null, false);
        this.layout_root = (ConstraintLayout) inflate.findViewById(R.id.layout_root);
        this.layout_device = (ConstraintLayout) inflate.findViewById(R.id.layout_device);
        this.recycler_device = (RecyclerView) inflate.findViewById(R.id.recycler_device);
        this.img_small = (ImageView) inflate.findViewById(R.id.img_small);
        this.img_drag = (ImageView) inflate.findViewById(R.id.img_drag);
        return inflate;
    }

    @Override // androidx.fragment.app.Fragment
    public void onDestroy() {
        super.onDestroy();
    }

    public void refreshDeviceInfo() {
        List<BleDeviceModel> list;
        List<BleDeviceModel> list2 = this.mList;
        if (list2 != null) {
            list2.clear();
        } else {
            this.mList = new ArrayList();
        }
        for (Map.Entry<String, BleProduct> entry : BleManager.getInstance(this.mContext).getBleProducts().entrySet()) {
            if (!entry.getKey().equals(MyApplication.getApplication().getDeviceAddress()) && entry.getValue() != null && entry.getValue().getmProductModel() != null && (list = this.mList) != null) {
                list.add(entry.getValue().getConnBleDevice());
            }
        }
        DeviceInfoAdapter deviceInfoAdapter = this.mAdapter;
        if (deviceInfoAdapter != null) {
            deviceInfoAdapter.notifyDataSetChanged();
        }
    }
}
