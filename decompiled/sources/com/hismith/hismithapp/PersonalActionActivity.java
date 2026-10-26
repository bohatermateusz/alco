package com.hismith.hismithapp;

import android.os.Bundle;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.fragment.app.Fragment;
import androidx.viewpager.widget.ViewPager;
import com.google.android.material.tabs.TabLayout;
import com.hismith.hismithapp.adapter.FragmentAdapter;
import com.hismith.hismithapp.fragment.PersonalCollectFragment;
import com.hismith.hismithapp.fragment.PersonalInteractionFragment;
import com.hismith.hismithapp.fragment.PersonalLikeFragment;
import com.hismith.hismithapp.fragment.PersonalNoticeFragment;
import com.hismith.hismithapp.fragment.PersonalReviewFragment;
import com.hismith.hismithapp.fragment.PersonalTopicFragment;
import com.hismith.hismithapp.utils.LanguageUtil;
import java.util.ArrayList;
import java.util.List;

/* loaded from: classes3.dex */
public class PersonalActionActivity extends BaseActivity implements View.OnClickListener, ViewPager.OnPageChangeListener {
    public static final String EXTRA_POSITION = "extra_position";
    private PersonalCollectFragment collectFragment;
    private List<Fragment> fragments;
    private PersonalLikeFragment likeFragment;
    private FragmentAdapter mFragmentAdapter;
    private TabLayout mTabLayout;
    private ImageView mTitleBack;
    private TextView mTitltName;
    private ViewPager mViewPager;
    private PersonalInteractionFragment mutuallyFragment;
    private PersonalNoticeFragment noticeFragment;
    private PersonalReviewFragment reviewFragment;
    private int tabPosition = 0;
    private PersonalTopicFragment topicFragment;

    private void init() {
        if (getIntent() != null) {
            this.tabPosition = getIntent().getIntExtra("extra_position", 0);
        }
        ArrayList arrayList = new ArrayList();
        arrayList.add(LanguageUtil.getLanguageByKey(this, "topicTitle"));
        arrayList.add(LanguageUtil.getLanguageByKey(this, "interactiveTitle"));
        arrayList.add(LanguageUtil.getLanguageByKey(this, "like"));
        arrayList.add(LanguageUtil.getLanguageByKey(this, "comment"));
        for (int i = 0; i < arrayList.size(); i++) {
            TabLayout tabLayout = this.mTabLayout;
            tabLayout.addTab(tabLayout.newTab().setText((CharSequence) arrayList.get(i)));
        }
        ArrayList arrayList2 = new ArrayList();
        this.fragments = arrayList2;
        PersonalTopicFragment personalTopicFragment = new PersonalTopicFragment();
        this.topicFragment = personalTopicFragment;
        arrayList2.add(personalTopicFragment);
        List<Fragment> list = this.fragments;
        PersonalInteractionFragment personalInteractionFragment = new PersonalInteractionFragment();
        this.mutuallyFragment = personalInteractionFragment;
        list.add(personalInteractionFragment);
        List<Fragment> list2 = this.fragments;
        PersonalLikeFragment personalLikeFragment = new PersonalLikeFragment();
        this.likeFragment = personalLikeFragment;
        list2.add(personalLikeFragment);
        List<Fragment> list3 = this.fragments;
        PersonalReviewFragment personalReviewFragment = new PersonalReviewFragment();
        this.reviewFragment = personalReviewFragment;
        list3.add(personalReviewFragment);
        this.mFragmentAdapter = new FragmentAdapter(getSupportFragmentManager(), this.fragments, arrayList);
        this.mViewPager.setOffscreenPageLimit(this.fragments.size());
        this.mViewPager.setAdapter(this.mFragmentAdapter);
        this.mViewPager.setCurrentItem(this.tabPosition);
        this.mTabLayout.setupWithViewPager(this.mViewPager);
        int tabCount = this.mTabLayout.getTabCount();
        int i2 = this.tabPosition;
        if (tabCount > i2) {
            changeTabTextView(this.mTabLayout.getTabAt(i2), true);
        }
    }

    private void initView() {
        this.mTitleBack = (ImageView) findViewById(R.id.title_img_back);
        this.mTitltName = (TextView) findViewById(R.id.title_txt_name);
        this.mTabLayout = (TabLayout) findViewById(R.id.personal_tab_layout);
        this.mViewPager = (ViewPager) findViewById(R.id.personal_viewpager);
    }

    private void setListener() {
        this.mTitleBack.setOnClickListener(this);
        this.mViewPager.addOnPageChangeListener(this);
        this.mTabLayout.addOnTabSelectedListener(new TabLayout.OnTabSelectedListener() { // from class: com.hismith.hismithapp.PersonalActionActivity.1
            @Override // com.google.android.material.tabs.TabLayout.BaseOnTabSelectedListener
            public void onTabReselected(TabLayout.Tab tab) {
            }

            @Override // com.google.android.material.tabs.TabLayout.BaseOnTabSelectedListener
            public void onTabSelected(TabLayout.Tab tab) {
                PersonalActionActivity.this.changeTabTextView(tab, true);
            }

            @Override // com.google.android.material.tabs.TabLayout.BaseOnTabSelectedListener
            public void onTabUnselected(TabLayout.Tab tab) {
                PersonalActionActivity.this.changeTabTextView(tab, false);
            }
        });
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

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        if (view.getId() == R.id.title_img_back) {
            finish();
        }
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, androidx.core.app.ComponentActivity, android.app.Activity
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        setContentView(R.layout.activity_personal_action);
        initView();
        init();
        setListener();
    }

    @Override // androidx.viewpager.widget.ViewPager.OnPageChangeListener
    public void onPageScrollStateChanged(int i) {
    }

    @Override // androidx.viewpager.widget.ViewPager.OnPageChangeListener
    public void onPageScrolled(int i, float f, int i2) {
    }

    @Override // androidx.viewpager.widget.ViewPager.OnPageChangeListener
    public void onPageSelected(int i) {
        this.tabPosition = i;
    }
}
