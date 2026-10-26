package com.hismith.hismithapp;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.hismith.hismithapp.adapter.LikeAdapter;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.http.RetrofitRequest;
import com.hismith.hismithapp.http.RxSubscribe;
import com.hismith.hismithapp.model.LikeUserModel;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import com.scwang.smart.refresh.layout.SmartRefreshLayout;
import com.scwang.smart.refresh.layout.api.RefreshLayout;
import com.scwang.smart.refresh.layout.listener.OnRefreshLoadMoreListener;
import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.schedulers.Schedulers;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/* loaded from: classes3.dex */
public class LikeUserActivity extends BaseActivity implements View.OnClickListener {
    private RetrofitRequest baseRequest;
    private LikeAdapter mAdapter;
    private FrameLayout mLayout;
    private List<LikeUserModel.User> mLikeList;
    private LinearLayoutManager mLinearLayoutManager;
    private RecyclerView mRecyclerView;
    private SmartRefreshLayout mRefreshLayout;
    private ImageView mTitleBack;
    private TextView mTitleName;
    private int reviewId = 0;
    private int page = 1;
    private int totalPage = 1;

    static /* synthetic */ int access$008(LikeUserActivity likeUserActivity) {
        int i = likeUserActivity.page;
        likeUserActivity.page = i + 1;
        return i;
    }

    private void init() {
        if (getIntent() != null) {
            this.reviewId = getIntent().getIntExtra("reviewId", 0);
        }
        this.baseRequest = new RetrofitRequest();
        LinearLayoutManager linearLayoutManager = new LinearLayoutManager(this);
        this.mLinearLayoutManager = linearLayoutManager;
        this.mRecyclerView.setLayoutManager(linearLayoutManager);
        this.mLikeList = new ArrayList();
        LikeAdapter likeAdapter = new LikeAdapter(this, this.mLikeList);
        this.mAdapter = likeAdapter;
        this.mRecyclerView.setAdapter(likeAdapter);
        this.mRefreshLayout.autoRefresh();
    }

    private void initView() {
        this.mLayout = (FrameLayout) findViewById(R.id.like_layout);
        this.mTitleBack = (ImageView) findViewById(R.id.title_img_back);
        this.mTitleName = (TextView) findViewById(R.id.title_txt_name);
        this.mRefreshLayout = (SmartRefreshLayout) findViewById(R.id.like_refreshLayout);
        this.mRecyclerView = (RecyclerView) findViewById(R.id.like_recycler_view);
    }

    private void setListener() {
        this.mTitleBack.setOnClickListener(this);
        this.mRefreshLayout.setOnRefreshLoadMoreListener(new OnRefreshLoadMoreListener() { // from class: com.hismith.hismithapp.LikeUserActivity.1
            @Override // com.scwang.smart.refresh.layout.listener.OnLoadMoreListener
            public void onLoadMore(RefreshLayout refreshLayout) {
                int i = LikeUserActivity.this.page;
                int i2 = LikeUserActivity.this.totalPage;
                LikeUserActivity likeUserActivity = LikeUserActivity.this;
                if (i >= i2) {
                    likeUserActivity.mRefreshLayout.finishLoadMore(1000);
                    return;
                }
                LikeUserActivity.access$008(likeUserActivity);
                LikeUserActivity likeUserActivity2 = LikeUserActivity.this;
                likeUserActivity2.topicReviewLikeUser(likeUserActivity2.page);
            }

            @Override // com.scwang.smart.refresh.layout.listener.OnRefreshListener
            public void onRefresh(RefreshLayout refreshLayout) {
                LikeUserActivity.this.page = 1;
                LikeUserActivity likeUserActivity = LikeUserActivity.this;
                likeUserActivity.topicReviewLikeUser(likeUserActivity.page);
            }
        });
        this.mAdapter.setOnItemCLickListener(new LikeAdapter.OnItemCLickListener() { // from class: com.hismith.hismithapp.LikeUserActivity.2
            @Override // com.hismith.hismithapp.adapter.LikeAdapter.OnItemCLickListener
            public void onItemClick(int i) {
                if (MyApplication.getApplication().getmUserModel() != null && ((LikeUserModel.User) LikeUserActivity.this.mLikeList.get(i)).getUserId() == MyApplication.getApplication().getmUserModel().getUserId()) {
                    LikeUserActivity.this.startActivity(new Intent(LikeUserActivity.this, (Class<?>) PersonalDetailsActivity2.class));
                } else {
                    Intent intent = new Intent(LikeUserActivity.this, (Class<?>) UserDetailsActivity2.class);
                    intent.putExtra("userId", ((LikeUserModel.User) LikeUserActivity.this.mLikeList.get(i)).getUserId());
                    LikeUserActivity.this.startActivity(intent);
                }
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void topicReviewLikeUser(int i) {
        this.baseRequest.getApi().topicReviewLikeUser((String) SharedPreferencesUtil.get(this, Constans.Token), Integer.valueOf(this.reviewId), Integer.valueOf(i)).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<LikeUserModel>(this, this.progressDialog, false) { // from class: com.hismith.hismithapp.LikeUserActivity.3
            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onComplete() {
                super.onComplete();
                LikeUserActivity.this.mRefreshLayout.finishLoadMore();
                LikeUserActivity.this.mRefreshLayout.finishRefresh();
            }

            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onError(Throwable th) {
                super.onError(th);
                LikeUserActivity.this.mRefreshLayout.finishLoadMore();
                LikeUserActivity.this.mRefreshLayout.finishRefresh();
            }

            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(LikeUserModel likeUserModel) {
                try {
                    if (likeUserModel.getSuccess() == 1) {
                        LikeUserActivity.this.page = likeUserModel.getPage();
                        LikeUserActivity.this.totalPage = likeUserModel.getTotalPage();
                        if (LikeUserActivity.this.page <= 1) {
                            LikeUserActivity.this.mLikeList.clear();
                        }
                        if (likeUserModel.getUsers() != null) {
                            LikeUserActivity.this.mLikeList.addAll(Arrays.asList(likeUserModel.getUsers()));
                            LikeUserActivity.this.mAdapter.notifyDataSetChanged();
                        }
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }
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
        setContentView(R.layout.activity_like_user);
        initView();
        init();
        setListener();
    }
}
