package com.hismith.hismithapp;

import android.os.Bundle;
import android.view.View;
import android.widget.ImageView;
import androidx.fragment.app.Fragment;
import androidx.viewpager.widget.ViewPager;
import com.hismith.hismithapp.adapter.FragmentAdapter;
import com.hismith.hismithapp.fragment.ModeCollectFragment;
import com.hismith.hismithapp.fragment.PersonalCollectFragment;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.view.SlidingTabLayout;
import java.util.ArrayList;
import java.util.List;

/* loaded from: classes3.dex */
public class PersonalCollectionActivity extends BaseActivity implements View.OnClickListener {
    public static final String EXTRA_POSITION = "extra_position";
    private List<Fragment> fragments;
    private FragmentAdapter mFragmentAdapter;
    private SlidingTabLayout mTabLayout;
    private ImageView mTitleBack;
    private ViewPager mViewPager;
    private int selectIndex = 0;

    private void init() {
        if (getIntent() != null) {
            this.selectIndex = getIntent().getIntExtra("extra_position", 0);
        }
        ArrayList arrayList = new ArrayList();
        arrayList.add(LanguageUtil.getLanguageByKey(this, "topicTitle"));
        arrayList.add(LanguageUtil.getLanguageByKey(this, "mode"));
        ArrayList arrayList2 = new ArrayList();
        this.fragments = arrayList2;
        arrayList2.add(new PersonalCollectFragment());
        this.fragments.add(new ModeCollectFragment());
        FragmentAdapter fragmentAdapter = new FragmentAdapter(getSupportFragmentManager(), this.fragments, arrayList);
        this.mFragmentAdapter = fragmentAdapter;
        this.mViewPager.setAdapter(fragmentAdapter);
        this.mViewPager.setCurrentItem(this.selectIndex);
        this.mTabLayout.setViewPager(this.mViewPager, (String[]) arrayList.toArray(new String[arrayList.size()]));
    }

    private void initView() {
        this.mTitleBack = (ImageView) findViewById(R.id.title_img_back);
        this.mTabLayout = (SlidingTabLayout) findViewById(R.id.collection_tab_layout);
        this.mViewPager = (ViewPager) findViewById(R.id.collection_view_pager);
    }

    private void setListener() {
        this.mTitleBack.setOnClickListener(this);
        this.mViewPager.addOnPageChangeListener(new ViewPager.OnPageChangeListener() { // from class: com.hismith.hismithapp.PersonalCollectionActivity.1
            @Override // androidx.viewpager.widget.ViewPager.OnPageChangeListener
            public void onPageScrollStateChanged(int i) {
            }

            @Override // androidx.viewpager.widget.ViewPager.OnPageChangeListener
            public void onPageScrolled(int i, float f, int i2) {
            }

            @Override // androidx.viewpager.widget.ViewPager.OnPageChangeListener
            public void onPageSelected(int i) {
                PersonalCollectionActivity.this.selectIndex = i;
            }
        });
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        if (view.getId() == R.id.title_img_back) {
            finish();
        }
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, androidx.core.app.ComponentActivity, android.app.Activity
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        setContentView(R.layout.activity_personal_collection);
        initView();
        init();
        setListener();
    }
}
