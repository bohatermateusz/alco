package com.hismith.hismithapp;

import android.os.Bundle;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.fragment.app.Fragment;
import androidx.viewpager.widget.ViewPager;
import com.hismith.hismithapp.adapter.FragmentAdapter;
import com.hismith.hismithapp.fragment.PersonalMessageFragment;
import com.hismith.hismithapp.utils.LanguageUtil;
import java.util.ArrayList;
import java.util.List;

/* loaded from: classes3.dex */
public class PersonalMessageActivity extends BaseActivity implements View.OnClickListener {
    public static final String EXTRA_POSITION = "extra_position";
    private List<Fragment> fragments;
    private FragmentAdapter mFragmentAdapter;
    private ImageView mTitleBack;
    private ViewPager mViewPager;
    private int selectIndex = 0;
    private TextView title_txt_name;

    private void init() {
        if (getIntent() != null) {
            this.selectIndex = getIntent().getIntExtra("extra_position", 0);
        }
        ArrayList arrayList = new ArrayList();
        arrayList.add(LanguageUtil.getLanguageByKey(this, "messageTitle"));
        ArrayList arrayList2 = new ArrayList();
        this.fragments = arrayList2;
        arrayList2.add(new PersonalMessageFragment());
        FragmentAdapter fragmentAdapter = new FragmentAdapter(getSupportFragmentManager(), this.fragments, arrayList);
        this.mFragmentAdapter = fragmentAdapter;
        this.mViewPager.setAdapter(fragmentAdapter);
        this.mViewPager.setCurrentItem(this.selectIndex);
    }

    private void initView() {
        this.mTitleBack = (ImageView) findViewById(R.id.title_img_back);
        this.title_txt_name = (TextView) findViewById(R.id.title_txt_name);
        this.mViewPager = (ViewPager) findViewById(R.id.message_view_pager);
        LanguageUtil.setText(this, this.title_txt_name, "messageTitle");
    }

    private void setListener() {
        this.mTitleBack.setOnClickListener(this);
        this.mViewPager.addOnPageChangeListener(new ViewPager.OnPageChangeListener() { // from class: com.hismith.hismithapp.PersonalMessageActivity.1
            @Override // androidx.viewpager.widget.ViewPager.OnPageChangeListener
            public void onPageScrollStateChanged(int i) {
            }

            @Override // androidx.viewpager.widget.ViewPager.OnPageChangeListener
            public void onPageScrolled(int i, float f, int i2) {
            }

            @Override // androidx.viewpager.widget.ViewPager.OnPageChangeListener
            public void onPageSelected(int i) {
                PersonalMessageActivity.this.selectIndex = i;
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
        setContentView(R.layout.activity_personal_message);
        initView();
        init();
        setListener();
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.appcompat.app.AppCompatActivity, androidx.fragment.app.FragmentActivity, android.app.Activity
    protected void onDestroy() {
        super.onDestroy();
    }

    @Override // androidx.appcompat.app.AppCompatActivity, androidx.fragment.app.FragmentActivity, android.app.Activity
    protected void onStart() {
        super.onStart();
    }
}
