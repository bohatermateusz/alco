package com.hismith.hismithapp.popup;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.bumptech.glide.Glide;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.model.TopicCategoriesModel;
import com.hismith.hismithapp.popup.CommonPopupWindow;
import com.hismith.hismithapp.utils.MeasureUtil;
import com.hismith.hismithapp.utils.ResUtil;
import com.hismith.hismithapp.view.decoration.DividerItemDecoration;
import java.util.List;

/* loaded from: classes3.dex */
public class PostAttachPopup2 {
    private ListAdapter mAdapter;
    private Context mContext;
    private LinearLayoutManager mLayoutManager;
    private List<TopicCategoriesModel.TopicType> mList;
    private RecyclerView mRecyclerView;
    private OnItemClickListener onItemClickListener;
    private CommonPopupWindow popupWindow;
    private View viewGroup;
    private int windowWidth;

    public class ListAdapter extends RecyclerView.Adapter<ViewHolder> {

        protected class ViewHolder extends RecyclerView.ViewHolder {
            private ImageView mIcon;
            private TextView mName;

            public ViewHolder(View view) {
                super(view);
                this.mIcon = (ImageView) view.findViewById(R.id.post_img_icon);
                this.mName = (TextView) view.findViewById(R.id.post_txt_name);
            }
        }

        public ListAdapter() {
        }

        @Override // androidx.recyclerview.widget.RecyclerView.Adapter
        public int getItemCount() {
            return PostAttachPopup2.this.mList.size();
        }

        @Override // androidx.recyclerview.widget.RecyclerView.Adapter
        public void onBindViewHolder(ViewHolder viewHolder, final int i) {
            Glide.with(PostAttachPopup2.this.mContext).load(((TopicCategoriesModel.TopicType) PostAttachPopup2.this.mList.get(i)).getIconUrl()).placeholder(ResUtil.getMipmapId(PostAttachPopup2.this.mContext, "ic_post_" + ((TopicCategoriesModel.TopicType) PostAttachPopup2.this.mList.get(i)).getIcon())).into(viewHolder.mIcon);
            viewHolder.mName.setText(((TopicCategoriesModel.TopicType) PostAttachPopup2.this.mList.get(i)).getTitle());
            viewHolder.itemView.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.popup.PostAttachPopup2.ListAdapter.1
                @Override // android.view.View.OnClickListener
                public void onClick(View view) {
                    if (PostAttachPopup2.this.onItemClickListener != null) {
                        PostAttachPopup2.this.onItemClickListener.onItemClick(((TopicCategoriesModel.TopicType) PostAttachPopup2.this.mList.get(i)).getId());
                        if (PostAttachPopup2.this.popupWindow != null) {
                            PostAttachPopup2.this.popupWindow.dismiss();
                        }
                    }
                }
            });
        }

        @Override // androidx.recyclerview.widget.RecyclerView.Adapter
        public ViewHolder onCreateViewHolder(ViewGroup viewGroup, int i) {
            return new ViewHolder(LayoutInflater.from(PostAttachPopup2.this.mContext).inflate(R.layout.item_post, viewGroup, false));
        }
    }

    public interface OnItemClickListener {
        void onItemClick(String str);
    }

    public PostAttachPopup2(Context context, List<TopicCategoriesModel.TopicType> list) {
        this.mContext = context;
        this.mList = list;
        initView();
        initData();
        setListener();
    }

    private void initData() {
        LinearLayoutManager linearLayoutManager = new LinearLayoutManager(this.mContext);
        this.mLayoutManager = linearLayoutManager;
        this.mRecyclerView.setLayoutManager(linearLayoutManager);
        this.mRecyclerView.addItemDecoration(new DividerItemDecoration(this.mContext, 1));
        ListAdapter listAdapter = new ListAdapter();
        this.mAdapter = listAdapter;
        this.mRecyclerView.setAdapter(listAdapter);
    }

    private void initView() {
        View inflate = LayoutInflater.from(this.mContext).inflate(R.layout.dialog_post_attach_list, (ViewGroup) null, false);
        this.viewGroup = inflate;
        this.mRecyclerView = (RecyclerView) inflate.findViewById(R.id.dialog_recycler_view);
    }

    private void setListener() {
    }

    public void setOnItemClickListener(OnItemClickListener onItemClickListener) {
        this.onItemClickListener = onItemClickListener;
    }

    public void setmList(List<TopicCategoriesModel.TopicType> list) {
        this.mList = list;
        ListAdapter listAdapter = this.mAdapter;
        if (listAdapter != null) {
            listAdapter.notifyDataSetChanged();
        }
    }

    public void showDownPop(View view) {
        CommonPopupWindow commonPopupWindow = this.popupWindow;
        if (commonPopupWindow == null || !commonPopupWindow.isShowing()) {
            this.popupWindow = new CommonPopupWindow.Builder(this.mContext).setView(this.viewGroup).setWidthAndHeight(-2, -2).setAnimationStyle(R.style.popmenu_animation).setOutsideTouchable(true).create();
            int[] iArr = new int[2];
            view.getLocationOnScreen(iArr);
            this.popupWindow.showAtLocation(view, 0, (iArr[0] - this.windowWidth) - MeasureUtil.dip2px(this.mContext, 90.0f), iArr[1] + view.getHeight() + MeasureUtil.dip2px(this.mContext, 5.0f));
        }
    }
}
