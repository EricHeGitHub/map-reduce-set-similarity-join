package comp9313.ass4;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.StringTokenizer;

import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;



public class SetSimJoin {
	// This mapper calculates the prefix and generate key-value pair of (itemID, setID)
	public static class PrefixMapper extends Mapper<Object, Text, Text, Text> {
		private Text ItemID = new Text();
		private Text setIDText = new Text();
		public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
			StringTokenizer itr = new StringTokenizer(value.toString(), "\n");
			Configuration conf = context.getConfiguration();
			double similarity = Double.parseDouble(conf.get("similarity"));
			// iterate over each item in the set
			while (itr.hasMoreTokens()) {
				String w = itr.nextToken();
				String[] wList = w.split("[\\s]+");
				
				List<String> elementList = Arrays.asList(wList).subList(1, wList.length);
				String elementString = StringUtils.join(elementList," ");
				String setID = String.valueOf(wList[0]);
				setIDText.set(setID + "," + elementString);
				int setLength = elementList.size();
				// calculate the prefix length: prefix = length of set - similarity * length of set + 1
				int prefixLength = (int) Math.floor(setLength - setLength * similarity + 1);
				if (prefixLength > 0){// if there are key-value pairs needed to be generated
					for(int i = 0; i < prefixLength; i++){
						ItemID.set(elementList.get(i));
						// Emit the key value pair
						context.write(ItemID, setIDText);
					}
				}
			}
		}
	}

	public static class SimilarityReducer extends Reducer<Text, Text, Text, NullWritable> {
		public void reduce(Text key, Iterable<Text> values, Context context)
				throws IOException, InterruptedException {
			Text pair = new Text();
			ArrayList<String> elements = new ArrayList<String>();
			Configuration conf = context.getConfiguration();
			double similarity = Double.parseDouble(conf.get("similarity"));
			for (Text val : values) {
				// collect all sets sharing the same item
				elements.add(val.toString());
			}
			for(int i = 0; i < elements.size(); i++){
				String[] setID_elementList_i = elements.get(i).split(","); 
				String setID_i = setID_elementList_i[0]; // set ID
				String elementList_i = setID_elementList_i[1]; // the list of items in the current set 		
				for(int j = i + 1; j < elements.size(); j++){
					// iterate all pairs of sets in the current value list to check similarity
					String[] setID_elementList_j = elements.get(j).split(",");
					String setID_j = setID_elementList_j[0];
					String elementList_j = setID_elementList_j[1];
					// The function used to calculate the similarity
					double sim = JaccardSimilarity(elementList_i, elementList_j);
					if(sim >= similarity){ // if the similarity between two sets is larger than the threshold
						String pairString = setID_i + "," + setID_j;
						pair.set(pairString +"\t" + String.valueOf(sim));
						// Emit the pairs of sets and their similarities
						context.write(pair, NullWritable.get());
					}
				}
			}
		}
	}
	
	public static class DeduplicateMapper extends Mapper<Object, Text, IntPair, Text> {
		Text simText = new Text();
		public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
			StringTokenizer itr = new StringTokenizer(value.toString(), "\n");
			while(itr.hasMoreTokens()){
				String w = itr.nextToken();
				String[] wList = w.split("\t");
				String pair = wList[0];
				String similarity = wList[1];
				String[] idPairString = pair.split(",");
				// Emit all pairs of with the first setID being smaller than the second setID
				int first = (Integer.parseInt(idPairString[0]) < Integer.parseInt(idPairString[1]))?
						Integer.parseInt(idPairString[0]): Integer.parseInt(idPairString[1]);
				int second = (Integer.parseInt(idPairString[0]) > Integer.parseInt(idPairString[1]))?
						Integer.parseInt(idPairString[0]): Integer.parseInt(idPairString[1]);
				IntPair intPair = new IntPair(first, second);
				simText.set(similarity);
				//Emit the key-value pair ((ID1, ID2), similarity)
				context.write(intPair, simText);
			}
		}
		
	}
	public static class DeduplicateCombiner extends Reducer<IntPair, Text, IntPair, Text> {
		Text output = new Text();
		// The combiner calculates the local result which ensures that the result from one mapper does not have
		// duplicated key-value pairs
		public void reduce(IntPair key, Iterable<Text> values, Context context) throws IOException, InterruptedException{
			ArrayList<String> sims = new ArrayList<String>();
			for (Text val : values) {
				sims.add(val.toString());
				break;
			}
			if(sims.size() > 0){
				output.set(sims.get(0));
				context.write(key, output);
			}
		}
	}
	public static class DeduplicateReducer extends Reducer<IntPair, Text, Text, NullWritable> {
		Text output = new Text();
		public void reduce(IntPair key, Iterable<Text> values, Context context) throws IOException, InterruptedException{
			ArrayList<String> sims = new ArrayList<String>();
			//Collect all (ID1,ID2) with the same similarity from different mappers
			for (Text val : values) {
				sims.add(val.toString());
				break;
			}
			
			if(sims.size() > 0){
				String s = "(" + key.getFirst() + "," + key.getSecond() + ")" + "\t" + sims.get(0);
				output.set(s);
				// Emit the deduplicated id pairs and similarities to the final output
				context.write(output, NullWritable.get());
			}
		}
	}
	
	public static double JaccardSimilarity(String s1, String s2){
		// use the definition of Jaccard Similarity as intersection(s1,s2) / union(s1, s2)
		Set<String> intersection = new HashSet<String>(Arrays.asList(s1.split("[\\s]+")));
		Set<String> union = new HashSet<String>(Arrays.asList(s1.split("[\\s]+")));
		Set<String> set2 = new HashSet<String>(Arrays.asList(s2.split("[\\s]+")));
		intersection.retainAll(set2);
		union.addAll(set2);
		double similarity = intersection.size() * 1.0 / union.size();
		return similarity;
	}
	public static void main(String[] args) throws Exception {
		String input = args[0];
		String output = args[1];
		String similarity = args[2];
		String maxSetID = "0";
		int reducerNumber = Integer.parseInt(args[3]);
		Configuration conf = new Configuration();
		conf.set("similarity", similarity);
		conf.set("maxSetID", maxSetID);
		Job job = Job.getInstance(conf, "Similarity Calculation");
		job.setJarByClass(SetSimJoin.class);
		job.setMapperClass(PrefixMapper.class);
		job.setNumReduceTasks(reducerNumber);
		//set the first map-reduce pair
		job.setReducerClass(SimilarityReducer.class);
		job.setMapOutputKeyClass(Text.class);
		job.setMapOutputValueClass(Text.class);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(NullWritable.class);
		FileInputFormat.addInputPath(job, new Path(input));
		FileOutputFormat.setOutputPath(job, new Path(output + "_intermediate"));
		job.waitForCompletion(true);
		
		// set the second map reduce pair
		job = Job.getInstance(conf, "Deduplication");
		job.setJarByClass(SetSimJoin.class);
		job.setMapperClass(DeduplicateMapper.class);
		// add the combiner to the distributed system
		job.setCombinerClass(DeduplicateCombiner.class);
		job.setNumReduceTasks(reducerNumber);
		job.setReducerClass(DeduplicateReducer.class);
		job.setMapOutputKeyClass(IntPair.class);
		job.setMapOutputValueClass(Text.class);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(NullWritable.class);
		FileInputFormat.addInputPath(job, new Path(output + "_intermediate"));
		FileOutputFormat.setOutputPath(job, new Path(output));
		job.waitForCompletion(true);
		System.exit(0);
		
	}
	
	public static class IntPair implements WritableComparable<IntPair> {
		private int first;
		private int second;

		// the default construction
		public IntPair() {
		}

		// using set function to initialize the two integers
		public IntPair(int first, int second) {
			set(first, second);
		}

		// set the value of length and number integers
		public void set(int first, int second) {
			this.first = first;
			this.second = second;
		}

		// return the value of length
		public int getFirst() {
			return this.first;
		}

		// return the value of number
		public int getSecond() {
			return this.second;
		}

		@Override
		// initialize numbers with input streams
		public void readFields(DataInput in) throws IOException {
			// TODO Auto-generated method stub
			this.first = in.readInt();
			this.second = in.readInt();
		}

		@Override
		// output the values of two integers to the output stream
		public void write(DataOutput out) throws IOException {
			// TODO Auto-generated method stub
			out.writeInt(this.first);
			out.writeInt(this.second);
		}
		

		@Override
		public int compareTo(IntPair o) {
			// TODO Auto-generated method stub
			if(this.first > o.first){
				return 1;
			}else if(this.first < o.first){
				return -1;
			}else{
				if(this.second > o.second){
					return 1;
				}else if(this.second < o.second){
					return -1;
				}else{
					return 0;
				}
			}
		}
		@Override
		public int hashCode(){
			String result = String.valueOf(this.first) + "," + String.valueOf(this.second);
			int hCode = result.hashCode();
			return hCode;
			
		}
	}
}
