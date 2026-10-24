from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
r=Path(__file__).resolve().parent;f=r/'figures';f.mkdir(exist_ok=True)
fig,ax=plt.subplots(figsize=(8,5.3));fig.subplots_adjust(left=.02,right=.98,top=.92,bottom=.02);ax.set_xlim(0,8);ax.set_ylim(0,5.2);ax.axis('off')
boxes=[(.1,3.7,'RGB input','B × 3 × S × S','INPUT'),(2.85,3.7,'Stem + layer1','B × 64 × S/4 × S/4','FROZEN + BN eval'),(5.6,3.7,'layer2','B × 128 × S/8 × S/8','FROZEN + BN eval'),(5.6,1.95,'layer3','B × 256 × S/16 × S/16','FROZEN + BN eval'),(2.85,1.95,'layer4','B × 512 × S/32 × S/32','TRAINABLE'),(.1,1.95,'AvgPool + FC','B × 512 → B × 2','TRAINABLE FC')]
for i,(x,y,title,dim,state) in enumerate(boxes):
 trainable=i>=4;c='#e3f4ea' if trainable else '#e7edf3';edge='#238451' if trainable else '#60778c'
 ax.add_patch(FancyBboxPatch((x,y),2.25,1.15,boxstyle='round,pad=0.035',facecolor=c,edgecolor=edge,lw=1.2));ax.text(x+1.125,y+.87,title,ha='center',va='center',fontsize=12,weight='bold');ax.text(x+1.125,y+.5,dim,ha='center',va='center',fontsize=10);ax.text(x+1.125,y+.14,state,ha='center',fontsize=9.5,color=edge)
for x1,y1,x2,y2 in [(2.4,4.27,2.77,4.27),(5.15,4.27,5.52,4.27),(6.72,3.64,6.72,3.17),(5.52,2.52,5.15,2.52),(2.77,2.52,2.4,2.52)]:ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops={'arrowstyle':'->','lw':1.4})
ax.text(.1,1.48,'S = 128: spatial sizes 32 → 16 → 8 → 4',fontsize=11)
ax.text(.1,1.12,'S = 224: spatial sizes 56 → 28 → 14 → 7',fontsize=11)
ax.text(.1,.76,'2 basic blocks per stage: y = ReLU(F(x) + shortcut(x)).',fontsize=10.5)
ax.text(.1,.4,'ImageNet-1K V1 backbone; new FC predicts cat (0) / dog (1).',fontsize=10.5)
ax.text(.1,.04,'11,177,538 parameters; 8,394,754 trainable (layer4 + FC).',fontsize=10.5)
ax.set_title('B classifier: partially fine-tuned ResNet18',fontsize=14,pad=8,loc='left')
fig.savefig(f/'resnet18_architecture_20epoch.png',dpi=220,facecolor='white');plt.close(fig)
