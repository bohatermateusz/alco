package com.hismith.hismithapp.dialog;

import android.content.Context;
import android.text.TextUtils;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import androidx.viewpager.widget.PagerAdapter;
import androidx.viewpager.widget.ViewPager;
import com.facebook.appevents.internal.ViewHierarchyConstants;
import com.google.android.material.tabs.TabLayout;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.http.RetrofitRequest;
import com.hismith.hismithapp.http.RxSubscribe;
import com.hismith.hismithapp.http.SocketManager;
import com.hismith.hismithapp.model.PlayModeListModel;
import com.hismith.hismithapp.model.PlayModeModel;
import com.hismith.hismithapp.surface_view.LineDataUtil;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import com.hismith.hismithapp.view.LinePointsView;
import com.lxj.xpopup.core.BottomPopupView;
import com.lxj.xpopup.util.XPopupUtils;
import com.scwang.smart.refresh.layout.SmartRefreshLayout;
import com.scwang.smart.refresh.layout.api.RefreshLayout;
import com.scwang.smart.refresh.layout.listener.OnRefreshLoadMoreListener;
import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.schedulers.Schedulers;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/* loaded from: classes3.dex */
public class PagerDrawerPopup extends BottomPopupView {
    private RetrofitRequest baseRequest;
    private Context mContext;
    private List<ListAdapter> mListAdapters;
    private List<List<PlayModeModel>> mModesList;
    private String[] mRequestIds;
    private String[] mRequestTypes;
    private ViewPager pager;
    private TabLayout tabLayout;
    private List<String> titles;
    private TextView txtTitle;

    private class ListAdapter extends RecyclerView.Adapter<ViewHolder> {
        private List<PlayModeModel> mList;

        class ViewHolder extends RecyclerView.ViewHolder {
            private LinePointsView line_points_view;
            private TextView txt_mode_name;
            private TextView txt_mode_time;

            public ViewHolder(View view) {
                super(view);
                this.txt_mode_name = (TextView) view.findViewById(R.id.txt_mode_name);
                this.line_points_view = (LinePointsView) view.findViewById(R.id.line_points_view);
                this.txt_mode_time = (TextView) view.findViewById(R.id.txt_mode_time);
            }
        }

        public ListAdapter(List<PlayModeModel> list) {
            this.mList = list;
        }

        @Override // androidx.recyclerview.widget.RecyclerView.Adapter
        public int getItemCount() {
            return this.mList.size();
        }

        public List<PlayModeModel> getList() {
            return this.mList;
        }

        @Override // androidx.recyclerview.widget.RecyclerView.Adapter
        public void onBindViewHolder(ViewHolder viewHolder, final int i) {
            String[] buildServerToClientData = LineDataUtil.buildServerToClientData(this.mList.get(i).getModePoints(), this.mList.get(i).getModeId());
            if (buildServerToClientData == null || TextUtils.isEmpty(buildServerToClientData[0])) {
                viewHolder.line_points_view.setData("", "");
            } else {
                viewHolder.line_points_view.setData(buildServerToClientData);
            }
            viewHolder.txt_mode_name.setText(this.mList.get(i).getModeName());
            viewHolder.txt_mode_time.setText(this.mList.get(i).getModeTime());
            viewHolder.itemView.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.dialog.PagerDrawerPopup.ListAdapter.1
                @Override // android.view.View.OnClickListener
                public void onClick(View view) {
                    SocketManager.getInstance().roomSetMode(((PlayModeModel) ListAdapter.this.mList.get(i)).getModeId(), 0);
                    PagerDrawerPopup.this.delayDismiss(100L);
                }
            });
        }

        @Override // androidx.recyclerview.widget.RecyclerView.Adapter
        public ViewHolder onCreateViewHolder(ViewGroup viewGroup, int i) {
            return new ViewHolder(LayoutInflater.from(PagerDrawerPopup.this.mContext).inflate(R.layout.adapter_mode_list, viewGroup, false));
        }
    }

    class PAdapter extends PagerAdapter {
        PAdapter() {
        }

        @Override // androidx.viewpager.widget.PagerAdapter
        public void destroyItem(ViewGroup viewGroup, int i, Object obj) {
            viewGroup.removeView((View) obj);
        }

        @Override // androidx.viewpager.widget.PagerAdapter
        public int getCount() {
            return PagerDrawerPopup.this.titles.size();
        }

        @Override // androidx.viewpager.widget.PagerAdapter
        public CharSequence getPageTitle(int i) {
            return (CharSequence) PagerDrawerPopup.this.titles.get(i);
        }

        @Override // androidx.viewpager.widget.PagerAdapter
        public Object instantiateItem(ViewGroup viewGroup, final int i) {
            LogUtil.e("instantiateItem：" + i + ", " + viewGroup);
            View inflate = LayoutInflater.from(viewGroup.getContext()).inflate(R.layout.view_refresh_layout, viewGroup, false);
            SmartRefreshLayout smartRefreshLayout = (SmartRefreshLayout) inflate.findViewById(R.id.refreshLayout);
            RecyclerView recyclerView = (RecyclerView) inflate.findViewById(R.id.recyclerView);
            recyclerView.setLayoutManager(new LinearLayoutManager(PagerDrawerPopup.this.mContext));
            ArrayList arrayList = new ArrayList();
            ListAdapter listAdapter = PagerDrawerPopup.this.new ListAdapter(arrayList);
            recyclerView.setAdapter(listAdapter);
            PagerDrawerPopup.this.mListAdapters.add(listAdapter);
            PagerDrawerPopup.this.mModesList.add(arrayList);
            smartRefreshLayout.setOnRefreshLoadMoreListener(new OnRefreshLoadMoreListener() { // from class: com.hismith.hismithapp.dialog.PagerDrawerPopup.PAdapter.1
                @Override // com.scwang.smart.refresh.layout.listener.OnLoadMoreListener
                public void onLoadMore(RefreshLayout refreshLayout) {
                    PagerDrawerPopup pagerDrawerPopup = PagerDrawerPopup.this;
                    String str = PagerDrawerPopup.this.mRequestTypes[i];
                    String[] strArr = PagerDrawerPopup.this.mRequestIds;
                    int i2 = i;
                    pagerDrawerPopup.playModeList(str, 0, strArr[i2], refreshLayout, i2);
                }

                @Override // com.scwang.smart.refresh.layout.listener.OnRefreshListener
                public void onRefresh(RefreshLayout refreshLayout) {
                    PagerDrawerPopup.this.mRequestIds[i] = "";
                    PagerDrawerPopup pagerDrawerPopup = PagerDrawerPopup.this;
                    String str = PagerDrawerPopup.this.mRequestTypes[i];
                    String[] strArr = PagerDrawerPopup.this.mRequestIds;
                    int i2 = i;
                    pagerDrawerPopup.playModeList(str, 1, strArr[i2], refreshLayout, i2);
                }
            });
            smartRefreshLayout.autoRefresh();
            viewGroup.addView(inflate);
            return inflate;
        }

        @Override // androidx.viewpager.widget.PagerAdapter
        public boolean isViewFromObject(View view, Object obj) {
            return view == obj;
        }
    }

    public PagerDrawerPopup(Context context) {
        super(context);
        this.mRequestTypes = new String[]{ViewHierarchyConstants.DIMENSION_TOP_KEY, "new", "my", "collect", "history"};
        this.mRequestIds = new String[5];
        this.mListAdapters = new ArrayList();
        this.mModesList = new ArrayList();
        this.mContext = context;
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void playModeList(String str, int i, final String str2, final RefreshLayout refreshLayout, final int i2) {
        this.baseRequest.getApi().playModeList((String) SharedPreferencesUtil.get(this.mContext, Constans.Token), str, MyApplication.getApplication().getDeviceCode(), i, str2).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<PlayModeListModel>(this.mContext, null, false) { // from class: com.hismith.hismithapp.dialog.PagerDrawerPopup.2
            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onComplete() {
                super.onComplete();
                refreshLayout.finishRefresh();
                refreshLayout.finishLoadMore();
            }

            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onError(Throwable th) {
                super.onError(th);
                refreshLayout.finishRefresh();
                refreshLayout.finishLoadMore();
            }

            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(PlayModeListModel playModeListModel) {
                if (playModeListModel.getSuccess() == 1) {
                    if (TextUtils.isEmpty(str2)) {
                        ((List) PagerDrawerPopup.this.mModesList.get(i2)).clear();
                        ((ListAdapter) PagerDrawerPopup.this.mListAdapters.get(i2)).notifyDataSetChanged();
                    }
                    if (!TextUtils.isEmpty(playModeListModel.getRequestId())) {
                        PagerDrawerPopup.this.mRequestIds[i2] = playModeListModel.getRequestId();
                    }
                    if (playModeListModel.getPlayModes() == null || playModeListModel.getPlayModes().length <= 0) {
                        return;
                    }
                    ((List) PagerDrawerPopup.this.mModesList.get(i2)).addAll(Arrays.asList(playModeListModel.getPlayModes()));
                    ((ListAdapter) PagerDrawerPopup.this.mListAdapters.get(i2)).notifyDataSetChanged();
                }
            }
        });
    }

    public void changeTabTextView(TabLayout.Tab tab, boolean z) {
        if (tab.getCustomView() == null) {
            tab.setCustomView(R.layout.tab_layout_text);
        }
        TextView textView = (TextView) tab.getCustomView().findViewById(android.R.id.text1);
        Context context = this.mContext;
        if (z) {
            textView.setTextAppearance(context, R.style.TabLayoutBoldTextSize);
        } else {
            textView.setTextAppearance(context, R.style.TabLayoutNormalTextSize);
        }
    }

    @Override // com.lxj.xpopup.core.BottomPopupView, com.lxj.xpopup.core.BasePopupView
    protected int getImplLayoutId() {
        return R.layout.dialog_pager_drawer;
    }

    @Override // com.lxj.xpopup.core.BasePopupView
    protected int getMaxHeight() {
        return (int) (XPopupUtils.getScreenHeight(getContext()) * 0.6f);
    }

    @Override // com.lxj.xpopup.core.BasePopupView
    protected void onCreate() {
        super.onCreate();
        this.baseRequest = new RetrofitRequest();
        ArrayList arrayList = new ArrayList();
        this.titles = arrayList;
        arrayList.add(LanguageUtil.getLanguageByKey(this.mContext, "modeMenuTop"));
        this.titles.add(LanguageUtil.getLanguageByKey(this.mContext, "modeMenuNew"));
        this.titles.add(LanguageUtil.getLanguageByKey(this.mContext, "modeMenuMy"));
        this.titles.add(LanguageUtil.getLanguageByKey(this.mContext, "modeMenuCollect"));
        this.titles.add(LanguageUtil.getLanguageByKey(this.mContext, "modeMenuHistory"));
        TextView textView = (TextView) findViewById(R.id.txtTitle);
        this.txtTitle = textView;
        LanguageUtil.setText(this.mContext, textView, "playMode");
        this.tabLayout = (TabLayout) findViewById(R.id.tabLayout);
        ViewPager viewPager = (ViewPager) findViewById(R.id.pager);
        this.pager = viewPager;
        viewPager.setOffscreenPageLimit(this.titles.size());
        this.pager.setAdapter(new PAdapter());
        this.tabLayout.setupWithViewPager(this.pager);
        if (this.tabLayout.getTabCount() > 0) {
            changeTabTextView(this.tabLayout.getTabAt(0), true);
        }
        this.tabLayout.addOnTabSelectedListener(new TabLayout.OnTabSelectedListener() { // from class: com.hismith.hismithapp.dialog.PagerDrawerPopup.1
            @Override // com.google.android.material.tabs.TabLayout.BaseOnTabSelectedListener
            public void onTabReselected(TabLayout.Tab tab) {
            }

            @Override // com.google.android.material.tabs.TabLayout.BaseOnTabSelectedListener
            public void onTabSelected(TabLayout.Tab tab) {
                PagerDrawerPopup.this.changeTabTextView(tab, true);
            }

            @Override // com.google.android.material.tabs.TabLayout.BaseOnTabSelectedListener
            public void onTabUnselected(TabLayout.Tab tab) {
                PagerDrawerPopup.this.changeTabTextView(tab, false);
            }
        });
    }

    @Override // com.lxj.xpopup.core.BasePopupView
    protected void onDismiss() {
        super.onDismiss();
    }

    @Override // com.lxj.xpopup.core.BasePopupView
    protected void onShow() {
        super.onShow();
    }
}
