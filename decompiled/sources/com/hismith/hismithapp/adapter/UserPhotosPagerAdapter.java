package com.hismith.hismithapp.adapter;

import android.content.Context;
import android.content.Intent;
import android.view.View;
import android.widget.ImageView;
import androidx.viewpager.widget.PagerAdapter;
import androidx.viewpager.widget.ViewPager;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.SexSettingsActivity;
import com.hismith.hismithapp.image_preview.ImagePreview;
import com.hismith.hismithapp.model.UserPhotosModel;
import com.hismith.hismithapp.utils.GlideImageUtil;
import java.util.List;

/* loaded from: classes3.dex */
public class UserPhotosPagerAdapter extends PagerAdapter {
    private List<String> imageList;
    private boolean isHasSexPhotos;
    private Context mContext;
    private List<View> pagerView;
    private UserPhotosModel[] photos;
    private int userId;
    private int viewSex;

    public UserPhotosPagerAdapter(Context context, List<View> list, int i, boolean z, List<String> list2, UserPhotosModel[] userPhotosModelArr, int i2) {
        this.mContext = context;
        this.pagerView = list;
        this.viewSex = i;
        this.isHasSexPhotos = z;
        this.imageList = list2;
        this.photos = userPhotosModelArr;
        this.userId = i2;
    }

    @Override // androidx.viewpager.widget.PagerAdapter
    public void destroyItem(View view, int i, Object obj) {
        if (i < this.pagerView.size()) {
            ((ViewPager) view).removeView(this.pagerView.get(i));
        }
    }

    @Override // androidx.viewpager.widget.PagerAdapter
    public int getCount() {
        return this.pagerView.size();
    }

    @Override // androidx.viewpager.widget.PagerAdapter
    public Object instantiateItem(View view, final int i) {
        ImageView imageView = (ImageView) this.pagerView.get(i).findViewById(R.id.item_details_image);
        imageView.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.UserPhotosPagerAdapter.1
            @Override // android.view.View.OnClickListener
            public void onClick(View view2) {
                if (UserPhotosPagerAdapter.this.isHasSexPhotos && UserPhotosPagerAdapter.this.viewSex == 0) {
                    UserPhotosPagerAdapter.this.mContext.startActivity(new Intent(UserPhotosPagerAdapter.this.mContext, (Class<?>) SexSettingsActivity.class));
                } else {
                    if (UserPhotosPagerAdapter.this.imageList == null || UserPhotosPagerAdapter.this.imageList.size() <= i) {
                        return;
                    }
                    ImagePreview.getInstance().setContext(UserPhotosPagerAdapter.this.mContext).setUserId(UserPhotosPagerAdapter.this.userId).setIndex(i).setImageList(UserPhotosPagerAdapter.this.imageList).setFolderName("hismith/image").setEnableDragClose(true).start();
                }
            }
        });
        GlideImageUtil.loadBlurredImage(this.mContext, this.photos[i].getUrl(), imageView, 0, 0, this.photos[i].getSex(), this.viewSex);
        ((ViewPager) view).addView(this.pagerView.get(i));
        return this.pagerView.get(i);
    }

    @Override // androidx.viewpager.widget.PagerAdapter
    public boolean isViewFromObject(View view, Object obj) {
        return view == obj;
    }
}
