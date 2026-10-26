package com.hismith.hismithapp.fragment;

import android.content.Intent;
import android.os.Bundle;
import android.text.TextUtils;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.hismith.hismithapp.BaseActivity;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.adapter.PlayModeAdapter;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.http.RetrofitRequest;
import com.hismith.hismithapp.http.RxSubscribe;
import com.hismith.hismithapp.model.PlayModeListModel;
import com.hismith.hismithapp.model.PlayModeModel;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import com.scwang.smart.refresh.layout.SmartRefreshLayout;
import com.scwang.smart.refresh.layout.api.RefreshLayout;
import com.scwang.smart.refresh.layout.listener.OnRefreshLoadMoreListener;
import io.reactivex.Observable;
import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.schedulers.Schedulers;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/* loaded from: classes3.dex */
public class ModeMyFragment extends LazyLoadFragment implements View.OnClickListener {
    private BaseActivity baseActivity;
    private RetrofitRequest baseRequest;
    private PlayModeAdapter mAdapter;
    private ImageView mImgEmpty;
    private LinearLayoutManager mLayoutManager;
    private List<PlayModeModel> mList;
    private RecyclerView mRecyclerView;
    private SmartRefreshLayout mRefreshLayout;
    private String mRequestId;
    private int[] popupIcon = {R.mipmap.ic_play_mode_item_rename, R.mipmap.ic_play_mode_item_edit, R.mipmap.ic_play_mode_item_copy, R.mipmap.ic_play_mode_item_share, R.mipmap.ic_play_mode_delete};
    private String[] popupText = {"rename", "edit", "copy", "modeShare", "delete"};
    private String mRequestType = "my";
    private int mFilterOnlyThisProduct = 0;
    private String mFilterModeName = null;
    private String mFilterModeTime = null;
    private int mFilterModeSpeed = 100;
    private String mFilterModePlayCount = null;
    private String mFilterPublishDate = null;
    private String mSortModeTime = null;
    private String mSortPlayCount = null;

    private void init() {
        this.baseRequest = new RetrofitRequest();
        LinearLayoutManager linearLayoutManager = new LinearLayoutManager(this.baseActivity);
        this.mLayoutManager = linearLayoutManager;
        this.mRecyclerView.setLayoutManager(linearLayoutManager);
        this.mList = new ArrayList();
        PlayModeAdapter playModeAdapter = new PlayModeAdapter(this.baseActivity, this.mList, this.mRequestType, true);
        this.mAdapter = playModeAdapter;
        this.mRecyclerView.setAdapter(playModeAdapter);
        this.mRefreshLayout.autoRefresh();
    }

    private void initView(View view) {
        this.mRefreshLayout = (SmartRefreshLayout) view.findViewById(R.id.mode_my_refreshLayout);
        this.mRecyclerView = (RecyclerView) view.findViewById(R.id.mode_my_recycler_view);
        this.mImgEmpty = (ImageView) view.findViewById(R.id.mode_my_img_empty);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void playModeList(String str, int i, final String str2) {
        Observable<PlayModeListModel> observeOn = this.baseRequest.getApi().playModeList((String) SharedPreferencesUtil.get(this.baseActivity, Constans.Token), str, MyApplication.getApplication().getDeviceCode(), i, str2, this.mSortModeTime, this.mSortPlayCount, this.mFilterOnlyThisProduct, this.mFilterModeName, this.mFilterModeTime, this.mFilterModeSpeed, this.mFilterModePlayCount, this.mFilterPublishDate).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread());
        BaseActivity baseActivity = this.baseActivity;
        observeOn.subscribe(new RxSubscribe<PlayModeListModel>(baseActivity, baseActivity.progressDialog, false) { // from class: com.hismith.hismithapp.fragment.ModeMyFragment.2
            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onComplete() {
                super.onComplete();
                ModeMyFragment.this.mRefreshLayout.finishRefresh();
                ModeMyFragment.this.mRefreshLayout.finishLoadMore();
            }

            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onError(Throwable th) {
                super.onError(th);
                ModeMyFragment.this.mRefreshLayout.finishRefresh();
                ModeMyFragment.this.mRefreshLayout.finishLoadMore();
            }

            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(PlayModeListModel playModeListModel) {
                if (playModeListModel.getSuccess() == 1) {
                    if (TextUtils.isEmpty(str2)) {
                        ModeMyFragment.this.mList.clear();
                        ModeMyFragment.this.mAdapter.notifyDataSetChanged();
                    }
                    ModeMyFragment.this.mRequestId = playModeListModel.getRequestId();
                    ModeMyFragment.this.mList.addAll(Arrays.asList(playModeListModel.getPlayModes()));
                    ModeMyFragment.this.mAdapter.notifyDataSetChanged();
                    int size = ModeMyFragment.this.mList.size();
                    ModeMyFragment modeMyFragment = ModeMyFragment.this;
                    if (size > 0) {
                        modeMyFragment.mImgEmpty.setVisibility(8);
                    } else {
                        modeMyFragment.mImgEmpty.setVisibility(0);
                    }
                }
            }
        });
    }

    private void setListener() {
        this.mRefreshLayout.setOnRefreshLoadMoreListener(new OnRefreshLoadMoreListener() { // from class: com.hismith.hismithapp.fragment.ModeMyFragment.1
            @Override // com.scwang.smart.refresh.layout.listener.OnLoadMoreListener
            public void onLoadMore(RefreshLayout refreshLayout) {
                ModeMyFragment modeMyFragment = ModeMyFragment.this;
                modeMyFragment.playModeList(modeMyFragment.mRequestType, 0, ModeMyFragment.this.mRequestId);
            }

            @Override // com.scwang.smart.refresh.layout.listener.OnRefreshListener
            public void onRefresh(RefreshLayout refreshLayout) {
                ModeMyFragment.this.mRequestId = "";
                ModeMyFragment modeMyFragment = ModeMyFragment.this;
                modeMyFragment.playModeList(modeMyFragment.mRequestType, 1, ModeMyFragment.this.mRequestId);
            }
        });
    }

    @Override // com.hismith.hismithapp.fragment.LazyLoadFragment
    protected void lazyLoad() {
        init();
        setListener();
    }

    @Override // androidx.fragment.app.Fragment
    public void onActivityResult(int i, int i2, Intent intent) {
        super.onActivityResult(i, i2, intent);
        LogUtil.e("创建模式界面返回requestCode：" + i + ", resultCode:" + i2);
        if (i == 17 && i2 == 17) {
            this.mRefreshLayout.autoRefresh();
        }
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
    }

    @Override // com.hismith.hismithapp.fragment.LazyLoadFragment, androidx.fragment.app.Fragment
    public void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        this.baseActivity = (BaseActivity) getActivity();
    }

    @Override // androidx.fragment.app.Fragment
    public View onCreateView(LayoutInflater layoutInflater, ViewGroup viewGroup, Bundle bundle) {
        View inflate = layoutInflater.inflate(R.layout.fragment_mode_my, (ViewGroup) null);
        initView(inflate);
        return inflate;
    }

    public void onRefresh(int i, String str, String str2, int i2, String str3, String str4, String str5, String str6) {
        this.mSortModeTime = str5;
        this.mSortPlayCount = str6;
        this.mFilterOnlyThisProduct = i;
        this.mFilterModeName = str;
        this.mFilterModeTime = str2;
        this.mFilterModeSpeed = i2;
        this.mFilterModePlayCount = str3;
        this.mFilterPublishDate = str4;
        if (this.isLoad) {
            refreshData();
        }
    }

    public void onRefresh(String str) {
        SmartRefreshLayout smartRefreshLayout;
        this.mFilterModeName = str;
        if (!this.isLoad || (smartRefreshLayout = this.mRefreshLayout) == null) {
            return;
        }
        smartRefreshLayout.autoRefresh();
    }

    public void refreshData() {
        SmartRefreshLayout smartRefreshLayout = this.mRefreshLayout;
        if (smartRefreshLayout != null) {
            smartRefreshLayout.autoRefresh();
        }
    }
}
