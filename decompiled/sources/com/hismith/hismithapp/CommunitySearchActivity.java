package com.hismith.hismithapp;

import android.os.Bundle;
import android.text.TextUtils;
import android.view.KeyEvent;
import android.view.View;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.TextView;
import androidx.activity.OnBackPressedCallback;
import androidx.fragment.app.Fragment;
import androidx.viewpager.widget.ViewPager;
import com.hismith.hismithapp.adapter.FragmentAdapter;
import com.hismith.hismithapp.fragment.CommunityTopicFragment;
import com.hismith.hismithapp.http.RetrofitRequest;
import com.hismith.hismithapp.model.TopicCategoriesModel;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import com.hismith.hismithapp.utils.SoftKeyboardUtil;
import com.hismith.hismithapp.view.MyViewPager;
import com.hismith.hismithapp.view.SlidingTabLayout;
import java.util.ArrayList;
import java.util.List;

/* loaded from: classes3.dex */
public class CommunitySearchActivity extends BaseActivity implements View.OnClickListener {
    private RetrofitRequest baseRequest;
    private List<Fragment> fragments;
    private EditText mEditSearch;
    private FragmentAdapter mFragmentAdapter;
    private FrameLayout mLayoutTabs;
    private SlidingTabLayout mTabLayout;
    private TextView mTxtCancel;
    private MyViewPager mViewPager;
    private int selectedIndex = 0;
    private String mSearchKey = "";

    private void init() {
        this.baseRequest = new RetrofitRequest();
        if (getIntent() != null) {
            this.selectedIndex = getIntent().getIntExtra("selectIndex", 0);
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void initCommunityMenu() {
        try {
            String str = (String) SharedPreferencesUtil.get(this, Constans.TopicCategpries, "");
            if (TextUtils.isEmpty(str)) {
                return;
            }
            ArrayList arrayList = new ArrayList();
            this.fragments = new ArrayList();
            TopicCategoriesModel topicCategoriesModel = (TopicCategoriesModel) this.gson.fromJson(str, TopicCategoriesModel.class);
            for (int i = 0; topicCategoriesModel.getMenu() != null && i < topicCategoriesModel.getMenu().length; i++) {
                String id = topicCategoriesModel.getMenu()[i].getId();
                arrayList.add(topicCategoriesModel.getMenu()[i].getTitle());
                this.fragments.add(CommunityTopicFragment.getInstance(id, this.mSearchKey));
            }
            if (arrayList.size() > 0) {
                this.mFragmentAdapter = new FragmentAdapter(getSupportFragmentManager(), this.fragments, arrayList);
                this.mViewPager.setOffscreenPageLimit(this.fragments.size());
                this.mViewPager.setAdapter(this.mFragmentAdapter);
                this.mTabLayout.setViewPager(this.mViewPager, (String[]) arrayList.toArray(new String[arrayList.size()]));
                this.mTabLayout.setCurrentTab(this.selectedIndex);
                this.mLayoutTabs.setVisibility(0);
                this.mViewPager.setVisibility(0);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void initView() {
        this.mEditSearch = (EditText) findViewById(R.id.community_edit_search);
        this.mTxtCancel = (TextView) findViewById(R.id.community_txt_cancel);
        this.mLayoutTabs = (FrameLayout) findViewById(R.id.community_layout_tabs);
        this.mTabLayout = (SlidingTabLayout) findViewById(R.id.community_tabs);
        this.mViewPager = (MyViewPager) findViewById(R.id.community_view_pager);
        LanguageUtil.setText(this, this.mTxtCancel, "cancel");
        this.mLayoutTabs.setVisibility(8);
        this.mViewPager.setVisibility(8);
    }

    private void setListener() {
        this.mTxtCancel.setOnClickListener(this);
        this.mEditSearch.setOnEditorActionListener(new TextView.OnEditorActionListener() { // from class: com.hismith.hismithapp.CommunitySearchActivity.2
            @Override // android.widget.TextView.OnEditorActionListener
            public boolean onEditorAction(TextView textView, int i, KeyEvent keyEvent) {
                if (i != 3) {
                    return false;
                }
                CommunitySearchActivity communitySearchActivity = CommunitySearchActivity.this;
                communitySearchActivity.mSearchKey = communitySearchActivity.mEditSearch.getText().toString().trim();
                if (!TextUtils.isEmpty(CommunitySearchActivity.this.mSearchKey)) {
                    CommunitySearchActivity.this.initCommunityMenu();
                }
                LogUtil.e("CommunityFragment发送" + CommunitySearchActivity.this.mSearchKey);
                SoftKeyboardUtil.hideSoftKeyboard(CommunitySearchActivity.this, textView);
                return true;
            }
        });
        this.mViewPager.addOnPageChangeListener(new ViewPager.OnPageChangeListener() { // from class: com.hismith.hismithapp.CommunitySearchActivity.3
            @Override // androidx.viewpager.widget.ViewPager.OnPageChangeListener
            public void onPageScrollStateChanged(int i) {
            }

            @Override // androidx.viewpager.widget.ViewPager.OnPageChangeListener
            public void onPageScrolled(int i, float f, int i2) {
            }

            @Override // androidx.viewpager.widget.ViewPager.OnPageChangeListener
            public void onPageSelected(int i) {
                CommunitySearchActivity.this.selectedIndex = i;
            }
        });
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        if (view.getId() == R.id.community_txt_cancel) {
            onBackPressed();
        }
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, androidx.core.app.ComponentActivity, android.app.Activity
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        getOnBackPressedDispatcher().addCallback(this, new OnBackPressedCallback(true) { // from class: com.hismith.hismithapp.CommunitySearchActivity.1
            @Override // androidx.activity.OnBackPressedCallback
            public void handleOnBackPressed() {
                CommunitySearchActivity.this.finish();
                CommunitySearchActivity.this.overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out);
            }
        });
        setContentView(R.layout.activity_community_search);
        initView();
        init();
        setListener();
    }

    @Override // android.app.Activity, android.view.Window.Callback
    public void onWindowFocusChanged(boolean z) {
        super.onWindowFocusChanged(z);
        this.mEditSearch.setFocusable(true);
        this.mEditSearch.setFocusableInTouchMode(true);
        this.mEditSearch.requestFocus();
        SoftKeyboardUtil.openKeyBoard(this, this.mEditSearch);
    }
}
