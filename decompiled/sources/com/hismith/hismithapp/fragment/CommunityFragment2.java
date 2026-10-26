package com.hismith.hismithapp.fragment;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.os.Bundle;
import android.text.TextUtils;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.fragment.app.Fragment;
import androidx.viewpager.widget.ViewPager;
import com.hismith.hismithapp.CommunitySearchActivity;
import com.hismith.hismithapp.MainActivity;
import com.hismith.hismithapp.PublishActivity;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.adapter.FragmentAdapter;
import com.hismith.hismithapp.http.RetrofitRequest;
import com.hismith.hismithapp.model.TopicCategoriesModel;
import com.hismith.hismithapp.popup.PostAttachPopup2;
import com.hismith.hismithapp.utils.AppUtil;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.MeasureUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import com.hismith.hismithapp.view.MyViewPager;
import com.hismith.hismithapp.view.SlidingTabLayout;
import com.lxj.xpopup.core.BasePopupView;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/* loaded from: classes3.dex */
public class CommunityFragment2 extends LazyLoadFragment implements View.OnClickListener {
    private RetrofitRequest baseRequest;
    private List<Fragment> fragments;
    private TextView mEditSearch;
    private FragmentAdapter mFragmentAdapter;
    private ImageView mImgPublish;
    private FrameLayout mLayout;
    private LinearLayout mLayoutSearch;
    private BasePopupView mPostPupup;
    private SlidingTabLayout mTabLayout;
    private MyViewPager mViewPager;
    private MainActivity mainActivity;
    private List<TopicCategoriesModel.TopicType> mPostList = new ArrayList();
    private int selectedIndex = 1;
    BroadcastReceiver mReceiver = new BroadcastReceiver() { // from class: com.hismith.hismithapp.fragment.CommunityFragment2.3
        @Override // android.content.BroadcastReceiver
        public void onReceive(Context context, Intent intent) {
            if (Constans.RECEIVER_TOPIC_MENU.equals(intent.getAction())) {
                CommunityFragment2.this.initCommunityMenu();
            }
        }
    };

    private void init() {
        this.baseRequest = new RetrofitRequest();
        initCommunityMenu();
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void initCommunityMenu() {
        try {
            String str = (String) SharedPreferencesUtil.get(this.mainActivity, Constans.TopicCategpries, "");
            if (TextUtils.isEmpty(str)) {
                return;
            }
            ArrayList arrayList = new ArrayList();
            this.fragments = new ArrayList();
            TopicCategoriesModel topicCategoriesModel = (TopicCategoriesModel) this.mainActivity.gson.fromJson(str, TopicCategoriesModel.class);
            for (int i = 0; topicCategoriesModel.getMenu() != null && i < topicCategoriesModel.getMenu().length; i++) {
                String id = topicCategoriesModel.getMenu()[i].getId();
                if ("2".equals(id)) {
                    this.selectedIndex = i;
                }
                arrayList.add(topicCategoriesModel.getMenu()[i].getTitle());
                this.fragments.add(CommunityTopicFragment.getInstance(id));
            }
            if (arrayList.size() > 0) {
                this.mFragmentAdapter = new FragmentAdapter(this.mainActivity.getSupportFragmentManager(), this.fragments, arrayList);
                this.mViewPager.setOffscreenPageLimit(this.fragments.size());
                this.mViewPager.setAdapter(this.mFragmentAdapter);
                this.mTabLayout.setViewPager(this.mViewPager, (String[]) arrayList.toArray(new String[arrayList.size()]));
                this.mTabLayout.setCurrentTab(this.selectedIndex);
            }
            this.mPostList = new ArrayList(Arrays.asList(topicCategoriesModel.getPost()));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void registerReceiver() {
        try {
            IntentFilter intentFilter = new IntentFilter();
            intentFilter.addAction(Constans.RECEIVER_TOPIC_MENU);
            AppUtil.registerReceiver(this.mainActivity, this.mReceiver, intentFilter);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void setListener() {
        this.mImgPublish.setOnClickListener(this);
        this.mEditSearch.setOnClickListener(this);
        this.mViewPager.addOnPageChangeListener(new ViewPager.OnPageChangeListener() { // from class: com.hismith.hismithapp.fragment.CommunityFragment2.1
            @Override // androidx.viewpager.widget.ViewPager.OnPageChangeListener
            public void onPageScrollStateChanged(int i) {
            }

            @Override // androidx.viewpager.widget.ViewPager.OnPageChangeListener
            public void onPageScrolled(int i, float f, int i2) {
            }

            @Override // androidx.viewpager.widget.ViewPager.OnPageChangeListener
            public void onPageSelected(int i) {
                CommunityFragment2.this.selectedIndex = i;
            }
        });
    }

    private void showPostGuide() {
        PostAttachPopup2 postAttachPopup2 = new PostAttachPopup2(this.mainActivity, this.mPostList);
        postAttachPopup2.setOnItemClickListener(new PostAttachPopup2.OnItemClickListener() { // from class: com.hismith.hismithapp.fragment.CommunityFragment2.2
            @Override // com.hismith.hismithapp.popup.PostAttachPopup2.OnItemClickListener
            public void onItemClick(String str) {
                Intent intent = new Intent(CommunityFragment2.this.mainActivity, (Class<?>) PublishActivity.class);
                intent.putExtra(PublishActivity.EXTRA_POST_ID, str);
                CommunityFragment2.this.startActivity(intent);
            }
        });
        postAttachPopup2.showDownPop(this.mImgPublish);
    }

    private void unregisterReceiver() {
        try {
            this.mainActivity.unregisterReceiver(this.mReceiver);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override // com.hismith.hismithapp.fragment.LazyLoadFragment
    protected void lazyLoad() {
        init();
        setListener();
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        int id = view.getId();
        if (id != R.id.community_edit_search) {
            if (id == R.id.community_img_publish) {
                showPostGuide();
            }
        } else {
            Intent intent = new Intent(this.mainActivity, (Class<?>) CommunitySearchActivity.class);
            intent.putExtra("selectIndex", this.mTabLayout.getCurrentTab());
            startActivity(intent);
            this.mainActivity.overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out);
        }
    }

    @Override // com.hismith.hismithapp.fragment.LazyLoadFragment, androidx.fragment.app.Fragment
    public void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        this.mainActivity = (MainActivity) getActivity();
    }

    @Override // androidx.fragment.app.Fragment
    public View onCreateView(LayoutInflater layoutInflater, ViewGroup viewGroup, Bundle bundle) {
        View inflate = layoutInflater.inflate(R.layout.fragment_community2, (ViewGroup) null);
        this.mLayout = (FrameLayout) inflate.findViewById(R.id.community_layout);
        this.mLayoutSearch = (LinearLayout) inflate.findViewById(R.id.community_layout_search);
        this.mEditSearch = (TextView) inflate.findViewById(R.id.community_edit_search);
        this.mImgPublish = (ImageView) inflate.findViewById(R.id.community_img_publish);
        this.mTabLayout = (SlidingTabLayout) inflate.findViewById(R.id.community_tabs);
        this.mViewPager = (MyViewPager) inflate.findViewById(R.id.community_view_pager);
        LanguageUtil.setHint(this.mainActivity, this.mEditSearch, "topContent");
        inflate.setPadding(0, MeasureUtil.getStatusBarHeight(this.mainActivity), 0, 0);
        registerReceiver();
        return inflate;
    }

    @Override // androidx.fragment.app.Fragment
    public void onDestroy() {
        super.onDestroy();
        unregisterReceiver();
    }
}
