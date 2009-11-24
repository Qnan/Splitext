using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Drawing;
using System.IO;

namespace png2ggg
{
   class Program
   {
      static void Main(string[] args)
      {
         Bitmap img = Image.FromFile(args[0]) as Bitmap;
         FileStream fs = new FileStream(args[1], FileMode.Create);
         BinaryWriter bw = new BinaryWriter(fs);
         int align = 128;
         int w = (img.Width + align - 1) / align * align;
         int h = (img.Height + align - 1) / align * align;
         bw.Write(w);
         bw.Write(h);
         bw.Write((int)img.HorizontalResolution);
         bw.Write((int)img.VerticalResolution);

         for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
               bw.Write((int)Math.Min(Math.Max(img.GetPixel(Math.Min(x, img.Width - 1), Math.Min(y, img.Height - 1)).GetBrightness()*255,0),255));
         bw.Close();
      }                 
   }
}
