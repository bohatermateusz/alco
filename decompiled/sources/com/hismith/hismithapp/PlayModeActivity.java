package com.hismith.hismithapp;

import android.app.Dialog;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.os.Build;
import android.os.Bundle;
import android.text.TextUtils;
import android.view.KeyEvent;
import android.view.LayoutInflater;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.view.Window;
import android.view.WindowManager;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.activity.result.ActivityResult;
import androidx.activity.result.ActivityResultCallback;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.Fragment;
import androidx.viewpager.widget.ViewPager;
import com.google.android.material.tabs.TabLayout;
import com.hismith.hismithapp.adapter.FragmentAdapter;
import com.hismith.hismithapp.bluetooth.BleManager;
import com.hismith.hismithapp.dialog.FilterModePopupDialog;
import com.hismith.hismithapp.fragment.ModeCollectFragment;
import com.hismith.hismithapp.fragment.ModeHistoryFragment;
import com.hismith.hismithapp.fragment.ModeMyFragment;
import com.hismith.hismithapp.fragment.ModeNewFragment;
import com.hismith.hismithapp.fragment.ModeTopFragment;
import com.hismith.hismithapp.product.ProductUtil;
import com.hismith.hismithapp.utils.AppUtil;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.SoftKeyboardUtil;
import com.hismith.hismithapp.utils.ToastUtil;
import com.lxj.xpopup.XPopup;
import com.lxj.xpopup.enums.PopupPosition;
import java.util.ArrayList;
import java.util.List;

/* loaded from: classes3.dex */
public class PlayModeActivity extends BaseActivity implements View.OnClickListener, View.OnTouchListener, ViewPager.OnPageChangeListener {
    public static final int REQUEST_CREATE_MODE = 17;
    public static final int WHAT_NOTIFYDATA = 1;
    private EditText edit_mode_name;
    private FilterModePopupDialog filterModePopupDialog;
    private List<Fragment> fragments;
    private ImageView img_filter;
    private ActivityResultLauncher mBluetoothLauncher;
    private String mFilterModeName;
    private String mFilterModePlayCount;
    private int mFilterModeSpeed;
    private String mFilterModeTime;
    private int mFilterOnlyThisProduct;
    private String mFilterPublishDate;
    private ImageView mImgAdd;
    private ImageView mImgBack;
    private ImageView mImgBluetooth;
    private ImageView mImgTabsLeft;
    private ImageView mImgTabsRight;
    private Dialog mSelectDialog;
    private String mSortModeTime;
    private String mSortPlayCount;
    private TabLayout mTabsLayout;
    private TextView mTxtName;
    private ViewPager mViewPager;
    private boolean isTabLayoutFirst = true;
    private BroadcastReceiver mReceiver = new BroadcastReceiver() { // from class: com.hismith.hismithapp.PlayModeActivity.8
        @Override // android.content.BroadcastReceiver
        public void onReceive(Context context, Intent intent) {
            if (intent.getAction().equals(Constans.RECEIVER_BLE_CONN_STATE)) {
                int intExtra = intent.getIntExtra("ble_state", 0);
                PlayModeActivity playModeActivity = PlayModeActivity.this;
                if (intExtra == 2) {
                    playModeActivity.mImgBluetooth.setImageDrawable(ContextCompat.getDrawable(PlayModeActivity.this, R.mipmap.ic_play_bluetooth_connect));
                } else {
                    playModeActivity.mImgBluetooth.setImageDrawable(ContextCompat.getDrawable(PlayModeActivity.this, R.mipmap.ic_play_bluetooth_disconnect));
                }
            }
        }
    };

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

    /* JADX INFO: Access modifiers changed from: private */
    public void dissSelectDialog() {
        Dialog dialog = this.mSelectDialog;
        if (dialog != null) {
            dialog.dismiss();
            this.mSelectDialog = null;
        }
    }

    private void init() {
        int status = BleManager.getInstance(this).getStatus();
        ImageView imageView = this.mImgBluetooth;
        if (status == 2) {
            imageView.setImageDrawable(ContextCompat.getDrawable(this, R.mipmap.ic_play_bluetooth_connect));
        } else {
            imageView.setImageDrawable(ContextCompat.getDrawable(this, R.mipmap.ic_play_bluetooth_disconnect));
        }
        ArrayList arrayList = new ArrayList();
        arrayList.add(LanguageUtil.getLanguageByKey(this, "modeMenuTop"));
        arrayList.add(LanguageUtil.getLanguageByKey(this, "modeMenuNew"));
        arrayList.add(LanguageUtil.getLanguageByKey(this, "modeMenuMy"));
        arrayList.add(LanguageUtil.getLanguageByKey(this, "modeMenuHistory"));
        arrayList.add(LanguageUtil.getLanguageByKey(this, "modeMenuCollect"));
        for (int i = 0; i < arrayList.size(); i++) {
            TabLayout tabLayout = this.mTabsLayout;
            tabLayout.addTab(tabLayout.newTab().setText((CharSequence) arrayList.get(i)));
        }
        ArrayList arrayList2 = new ArrayList();
        this.fragments = arrayList2;
        arrayList2.add(new ModeTopFragment());
        this.fragments.add(new ModeNewFragment());
        this.fragments.add(new ModeMyFragment());
        this.fragments.add(new ModeHistoryFragment());
        this.fragments.add(new ModeCollectFragment());
        FragmentAdapter fragmentAdapter = new FragmentAdapter(getSupportFragmentManager(), this.fragments, arrayList);
        this.mViewPager.setOffscreenPageLimit(this.fragments.size());
        this.mViewPager.setAdapter(fragmentAdapter);
        this.mTabsLayout.setupWithViewPager(this.mViewPager);
        if (this.mTabsLayout.getTabCount() > 0) {
            changeTabTextView(this.mTabsLayout.getTabAt(0), true);
        }
        this.filterModePopupDialog = new FilterModePopupDialog(this, new FilterModePopupDialog.OnFilterChangeListener() { // from class: com.hismith.hismithapp.PlayModeActivity.1
            @Override // com.hismith.hismithapp.dialog.FilterModePopupDialog.OnFilterChangeListener
            public void onFilterChange(int i2, String str, String str2, int i3, String str3, String str4, String str5, String str6) {
                String str7 = str4;
                String str8 = str5;
                String str9 = str6;
                LogUtil.e("filterOnlyThisProduct:" + i2 + ", filterModeName:" + str + ", filterModeTime:" + str2 + ", filterModeSpeed:" + i3 + ", filterModePlayCount:" + str3 + ", filterPublishDate:" + str7 + ", sortModeTime:" + str8 + ", sortPlayCount:" + str9);
                PlayModeActivity.this.mFilterOnlyThisProduct = i2;
                PlayModeActivity.this.mFilterModeName = str;
                PlayModeActivity.this.mFilterModeTime = str2;
                PlayModeActivity.this.mFilterModeSpeed = i3;
                PlayModeActivity.this.mFilterModePlayCount = str3;
                PlayModeActivity.this.mFilterPublishDate = str7;
                PlayModeActivity.this.mSortModeTime = str8;
                PlayModeActivity.this.mSortPlayCount = str9;
                if (PlayModeActivity.this.mFilterOnlyThisProduct == 0 && TextUtils.isEmpty(PlayModeActivity.this.mFilterModeName) && PlayModeActivity.this.mFilterModeSpeed == 100 && ((TextUtils.isEmpty(PlayModeActivity.this.mFilterModeTime) || "-".equals(PlayModeActivity.this.mFilterModeTime)) && ((TextUtils.isEmpty(PlayModeActivity.this.mFilterModePlayCount) || "-".equals(PlayModeActivity.this.mFilterModePlayCount)) && TextUtils.isEmpty(PlayModeActivity.this.mFilterPublishDate) && TextUtils.isEmpty(PlayModeActivity.this.mSortModeTime) && TextUtils.isEmpty(PlayModeActivity.this.mSortPlayCount)))) {
                    PlayModeActivity.this.img_filter.setImageResource(R.mipmap.ic_mode_filter);
                } else {
                    PlayModeActivity.this.img_filter.setImageResource(R.mipmap.ic_mode_filter_selected);
                }
                PlayModeActivity.this.edit_mode_name.setText(PlayModeActivity.this.mFilterModeName);
                if (!TextUtils.isEmpty(PlayModeActivity.this.mFilterModeName)) {
                    PlayModeActivity.this.edit_mode_name.setSelection(PlayModeActivity.this.mFilterModeName.length());
                }
                int i4 = 0;
                while (i4 < PlayModeActivity.this.fragments.size()) {
                    try {
                        Fragment fragment = (Fragment) PlayModeActivity.this.fragments.get(i4);
                        if (fragment != null) {
                            if (fragment instanceof ModeTopFragment) {
                                ((ModeTopFragment) fragment).onRefresh(i2, str, str2, i3, str3, str7, str8, str9);
                            } else if (fragment instanceof ModeNewFragment) {
                                ((ModeNewFragment) fragment).onRefresh(i2, str, str2, i3, str3, str4, str5, str6);
                            } else if (fragment instanceof ModeMyFragment) {
                                ((ModeMyFragment) fragment).onRefresh(i2, str, str2, i3, str3, str4, str5, str6);
                            } else if (fragment instanceof ModeHistoryFragment) {
                                ((ModeHistoryFragment) fragment).onRefresh(i2, str, str2, i3, str3, str4, str5, str6);
                            } else if (fragment instanceof ModeCollectFragment) {
                                ((ModeCollectFragment) fragment).onRefresh(i2, str, str2, i3, str3, str4, str5, str6);
                            }
                        }
                        i4++;
                        str7 = str4;
                        str8 = str5;
                        str9 = str6;
                    } catch (Exception e) {
                        e.printStackTrace();
                        return;
                    }
                }
            }
        });
        initActivityResult();
        makeFilter();
    }

    private void initActivityResult() {
        this.mBluetoothLauncher = registerForActivityResult(new ActivityResultContracts.StartActivityForResult(), new ActivityResultCallback<ActivityResult>() { // from class: com.hismith.hismithapp.PlayModeActivity.4
            @Override // androidx.activity.result.ActivityResultCallback
            public void onActivityResult(ActivityResult activityResult) {
                if (activityResult.getResultCode() == -1) {
                    LogUtil.e("蓝牙打开");
                    PlayModeActivity.this.startActivity(new Intent(PlayModeActivity.this, (Class<?>) BluetootListActivity.class));
                } else if (activityResult.getResultCode() == 0) {
                    LogUtil.e("蓝牙打不开");
                }
            }
        });
    }

    private void initView() {
        this.mImgBack = (ImageView) findViewById(R.id.title_img_back);
        this.mImgBluetooth = (ImageView) findViewById(R.id.mode_img_bluetooth);
        this.mTxtName = (TextView) findViewById(R.id.title_txt_name);
        this.edit_mode_name = (EditText) findViewById(R.id.edit_mode_name);
        this.img_filter = (ImageView) findViewById(R.id.img_filter);
        this.mTabsLayout = (TabLayout) findViewById(R.id.mode_tabs);
        this.mImgTabsLeft = (ImageView) findViewById(R.id.mode_img_tabs_left);
        this.mImgTabsRight = (ImageView) findViewById(R.id.mode_img_tabs_right);
        this.mViewPager = (ViewPager) findViewById(R.id.mode_viewPager);
        this.mImgAdd = (ImageView) findViewById(R.id.mode_my_img_add);
        LanguageUtil.setText(this, this.mTxtName, "playMode");
    }

    private void makeFilter() {
        IntentFilter intentFilter = new IntentFilter();
        intentFilter.addAction(Constans.RECEIVER_BLE_CONN_STATE);
        AppUtil.registerReceiver(this, this.mReceiver, intentFilter);
    }

    private void scrollHideImg() {
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void searchNameForModeList(String str) {
        this.mFilterModeName = str;
        if (this.mFilterOnlyThisProduct == 0 && TextUtils.isEmpty(str) && this.mFilterModeSpeed == 100 && ((TextUtils.isEmpty(this.mFilterModeTime) || "-".equals(this.mFilterModeTime)) && ((TextUtils.isEmpty(this.mFilterModePlayCount) || "-".equals(this.mFilterModePlayCount)) && TextUtils.isEmpty(this.mFilterPublishDate) && TextUtils.isEmpty(this.mSortModeTime) && TextUtils.isEmpty(this.mSortPlayCount)))) {
            this.img_filter.setImageResource(R.mipmap.ic_mode_filter);
        } else {
            this.img_filter.setImageResource(R.mipmap.ic_mode_filter_selected);
        }
        for (int i = 0; i < this.fragments.size(); i++) {
            try {
                Fragment fragment = this.fragments.get(i);
                if (fragment != null) {
                    if (fragment instanceof ModeTopFragment) {
                        ((ModeTopFragment) fragment).onRefresh(str);
                    } else if (fragment instanceof ModeNewFragment) {
                        ((ModeNewFragment) fragment).onRefresh(str);
                    } else if (fragment instanceof ModeMyFragment) {
                        ((ModeMyFragment) fragment).onRefresh(str);
                    } else if (fragment instanceof ModeHistoryFragment) {
                        ((ModeHistoryFragment) fragment).onRefresh(str);
                    } else if (fragment instanceof ModeCollectFragment) {
                        ((ModeCollectFragment) fragment).onRefresh(str);
                    }
                }
            } catch (Exception e) {
                e.printStackTrace();
                return;
            }
        }
    }

    private void setListener() {
        this.mImgBack.setOnClickListener(this);
        this.mImgBluetooth.setOnClickListener(this);
        this.img_filter.setOnClickListener(this);
        this.mTabsLayout.setOnTouchListener(this);
        this.mViewPager.addOnPageChangeListener(this);
        this.mImgAdd.setOnClickListener(this);
        this.mTabsLayout.addOnTabSelectedListener(new TabLayout.OnTabSelectedListener() { // from class: com.hismith.hismithapp.PlayModeActivity.2
            @Override // com.google.android.material.tabs.TabLayout.BaseOnTabSelectedListener
            public void onTabReselected(TabLayout.Tab tab) {
            }

            @Override // com.google.android.material.tabs.TabLayout.BaseOnTabSelectedListener
            public void onTabSelected(TabLayout.Tab tab) {
                LogUtil.e("onTabSelected");
                PlayModeActivity.this.changeTabTextView(tab, true);
            }

            @Override // com.google.android.material.tabs.TabLayout.BaseOnTabSelectedListener
            public void onTabUnselected(TabLayout.Tab tab) {
                LogUtil.e("onTabUnselected");
                PlayModeActivity.this.changeTabTextView(tab, false);
            }
        });
        this.edit_mode_name.setOnEditorActionListener(new TextView.OnEditorActionListener() { // from class: com.hismith.hismithapp.PlayModeActivity.3
            @Override // android.widget.TextView.OnEditorActionListener
            public boolean onEditorAction(TextView textView, int i, KeyEvent keyEvent) {
                if (i != 3) {
                    return false;
                }
                String trim = PlayModeActivity.this.edit_mode_name.getText().toString().trim();
                PlayModeActivity.this.searchNameForModeList(trim);
                if (PlayModeActivity.this.filterModePopupDialog != null) {
                    PlayModeActivity.this.filterModePopupDialog.setFilterModeName(trim);
                }
                SoftKeyboardUtil.hideSoftKeyboard(PlayModeActivity.this, textView);
                return true;
            }
        });
    }

    private void showSelectDialog() {
        Dialog dialog = this.mSelectDialog;
        if (dialog == null || !dialog.isShowing()) {
            dissSelectDialog();
            this.mSelectDialog = new Dialog(this, R.style.MyDialog);
            View inflate = LayoutInflater.from(this).inflate(R.layout.dialog_select_mode, (ViewGroup) null, false);
            LinearLayout linearLayout = (LinearLayout) inflate.findViewById(R.id.layout_professional);
            LinearLayout linearLayout2 = (LinearLayout) inflate.findViewById(R.id.layout_standard);
            ImageView imageView = (ImageView) inflate.findViewById(R.id.img_close);
            LanguageUtil.setText(this, (TextView) inflate.findViewById(R.id.txt_professional), "professional");
            LanguageUtil.setText(this, (TextView) inflate.findViewById(R.id.txt_standard), "standard");
            linearLayout.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.PlayModeActivity.5
                @Override // android.view.View.OnClickListener
                public void onClick(View view) {
                    if (!ProductUtil.hasSType()) {
                        PlayModeActivity playModeActivity = PlayModeActivity.this;
                        ToastUtil.showShortToast(playModeActivity, LanguageUtil.getLanguageByKey(playModeActivity, "deviceNotCreateMode"));
                    } else {
                        PlayModeActivity.this.startActivityForResult(new Intent(PlayModeActivity.this, (Class<?>) PlayModeCreateActivity.class), 17);
                        PlayModeActivity.this.dissSelectDialog();
                    }
                }
            });
            linearLayout2.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.PlayModeActivity.6
                @Override // android.view.View.OnClickListener
                public void onClick(View view) {
                    if (!ProductUtil.hasSType()) {
                        PlayModeActivity playModeActivity = PlayModeActivity.this;
                        ToastUtil.showShortToast(playModeActivity, LanguageUtil.getLanguageByKey(playModeActivity, "deviceNotCreateMode"));
                    } else {
                        PlayModeActivity.this.startActivityForResult(new Intent(PlayModeActivity.this, (Class<?>) PlayModeCreateActivity2.class), 17);
                        PlayModeActivity.this.dissSelectDialog();
                    }
                }
            });
            imageView.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.PlayModeActivity.7
                @Override // android.view.View.OnClickListener
                public void onClick(View view) {
                    PlayModeActivity.this.dissSelectDialog();
                }
            });
            this.mSelectDialog.setContentView(inflate);
            this.mSelectDialog.setCanceledOnTouchOutside(true);
            Window window = this.mSelectDialog.getWindow();
            if (window != null) {
                WindowManager.LayoutParams attributes = window.getAttributes();
                attributes.width = (int) (getWindowManager().getDefaultDisplay().getWidth() * 0.8d);
                window.setAttributes(attributes);
            }
            this.mSelectDialog.show();
        }
    }

    private void unRegisterReceiver() {
        try {
            unregisterReceiver(this.mReceiver);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void changeTabTextView(TabLayout.Tab tab, boolean z) {
        if (tab.getCustomView() == null) {
            tab.setCustomView(R.layout.tab_layout_text);
        }
        TextView textView = (TextView) tab.getCustomView().findViewById(android.R.id.text1);
        if (z) {
            textView.setTextAppearance(this, R.style.TabLayoutBoldTextSize);
        } else {
            textView.setTextAppearance(this, R.style.TabLayoutNormalTextSize);
        }
    }

    @Override // androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, android.app.Activity
    protected void onActivityResult(int i, int i2, Intent intent) {
        super.onActivityResult(i, i2, intent);
        LogUtil.e("模式界面返回requestCode：" + i + ", resultCode:" + i2);
        if (i == 17 && i2 == 17) {
            try {
                this.mViewPager.setCurrentItem(2);
                ((ModeMyFragment) this.fragments.get(2)).refreshData();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        int id = view.getId();
        if (id == R.id.title_img_back) {
            finish();
            return;
        }
        if (id == R.id.mode_img_bluetooth) {
            if (bluetoothIsEnabled()) {
                startActivity(new Intent(this, (Class<?>) BluetootListActivity.class));
            }
        } else if (id == R.id.img_filter) {
            new XPopup.Builder(this).isDestroyOnDismiss(false).popupPosition(PopupPosition.Right).hasStatusBarShadow(true).autoFocusEditText(false).autoOpenSoftInput(false).asCustom(this.filterModePopupDialog).show();
        } else if (id == R.id.mode_my_img_add) {
            showSelectDialog();
        }
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, androidx.core.app.ComponentActivity, android.app.Activity
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        setContentView(R.layout.activity_play_mode);
        initView();
        init();
        setListener();
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.appcompat.app.AppCompatActivity, androidx.fragment.app.FragmentActivity, android.app.Activity
    protected void onDestroy() {
        super.onDestroy();
        unRegisterReceiver();
    }

    @Override // androidx.viewpager.widget.ViewPager.OnPageChangeListener
    public void onPageScrollStateChanged(int i) {
        if (i == 0) {
            scrollHideImg();
        }
    }

    @Override // androidx.viewpager.widget.ViewPager.OnPageChangeListener
    public void onPageScrolled(int i, float f, int i2) {
        if (this.isTabLayoutFirst) {
            this.isTabLayoutFirst = false;
            scrollHideImg();
        }
    }

    @Override // androidx.viewpager.widget.ViewPager.OnPageChangeListener
    public void onPageSelected(int i) {
    }

    @Override // android.view.View.OnTouchListener
    public boolean onTouch(View view, MotionEvent motionEvent) {
        if (view.getId() != R.id.mode_tabs || motionEvent.getAction() != 2) {
            return false;
        }
        scrollHideImg();
        return false;
    }
}
