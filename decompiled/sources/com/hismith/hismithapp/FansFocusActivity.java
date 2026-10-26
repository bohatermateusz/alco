package com.hismith.hismithapp;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.fragment.app.Fragment;
import androidx.viewpager.widget.ViewPager;
import com.hismith.hismithapp.adapter.FragmentAdapter;
import com.hismith.hismithapp.fragment.PersonalFansFragment;
import com.hismith.hismithapp.fragment.PersonalFocusFragment;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.view.SlidingTabLayout;
import java.util.ArrayList;
import java.util.List;

/* loaded from: classes3.dex */
public class FansFocusActivity extends BaseActivity implements View.OnClickListener {
    private int fansMode = 0;
    private List<Fragment> fragments;
    private FragmentAdapter mFragmentAdapter;
    private SlidingTabLayout mTabs;
    private ImageView mTitleImgAdd;
    private ImageView mTitleImgBack;
    private TextView mTitleTxtName;
    private ViewPager mViewPager;

    private void init() {
        if (getIntent() != null) {
            this.fansMode = getIntent().getIntExtra("fansMode", 0);
        }
        ArrayList arrayList = new ArrayList();
        arrayList.add(LanguageUtil.getLanguageByKey(this, "fansCount"));
        arrayList.add(LanguageUtil.getLanguageByKey(this, "following"));
        ArrayList arrayList2 = new ArrayList();
        this.fragments = arrayList2;
        arrayList2.add(new PersonalFansFragment());
        this.fragments.add(new PersonalFocusFragment());
        FragmentAdapter fragmentAdapter = new FragmentAdapter(getSupportFragmentManager(), this.fragments, arrayList);
        this.mFragmentAdapter = fragmentAdapter;
        this.mViewPager.setAdapter(fragmentAdapter);
        this.mTabs.setViewPager(this.mViewPager, (String[]) arrayList.toArray(new String[arrayList.size()]));
        int i = this.fansMode;
        TextView textView = this.mTitleTxtName;
        if (i == 1) {
            LanguageUtil.setText(this, textView, "fansCount");
            this.mViewPager.setCurrentItem(0);
        } else {
            LanguageUtil.setText(this, textView, "following");
            this.mViewPager.setCurrentItem(1);
        }
    }

    private void initView() {
        this.mTitleImgBack = (ImageView) findViewById(R.id.title_img_back);
        this.mTitleTxtName = (TextView) findViewById(R.id.title_txt_name);
        this.mTitleImgAdd = (ImageView) findViewById(R.id.title_img_add);
        this.mTabs = (SlidingTabLayout) findViewById(R.id.fans_focus_tabs);
        this.mViewPager = (ViewPager) findViewById(R.id.fans_focus_viewpager);
    }

    private void setListener() {
        this.mTitleImgBack.setOnClickListener(this);
        this.mTitleImgAdd.setOnClickListener(this);
        this.mViewPager.addOnPageChangeListener(new ViewPager.OnPageChangeListener() { // from class: com.hismith.hismithapp.FansFocusActivity.1
            @Override // androidx.viewpager.widget.ViewPager.OnPageChangeListener
            public void onPageScrollStateChanged(int i) {
            }

            @Override // androidx.viewpager.widget.ViewPager.OnPageChangeListener
            public void onPageScrolled(int i, float f, int i2) {
            }

            @Override // androidx.viewpager.widget.ViewPager.OnPageChangeListener
            public void onPageSelected(int i) {
                if (i == 0) {
                    FansFocusActivity fansFocusActivity = FansFocusActivity.this;
                    LanguageUtil.setText(fansFocusActivity, fansFocusActivity.mTitleTxtName, "fansCount");
                } else if (i == 1) {
                    FansFocusActivity fansFocusActivity2 = FansFocusActivity.this;
                    LanguageUtil.setText(fansFocusActivity2, fansFocusActivity2.mTitleTxtName, "following");
                }
            }
        });
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        int id = view.getId();
        if (id == R.id.title_img_back) {
            finish();
        } else if (id == R.id.title_img_add) {
            startActivity(new Intent(this, (Class<?>) AddFriendActivity.class));
        }
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, androidx.core.app.ComponentActivity, android.app.Activity
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        setContentView(R.layout.activity_fans_focus);
        initView();
        init();
        setListener();
    }
}
