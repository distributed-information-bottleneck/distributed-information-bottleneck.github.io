#include <iostream>
#include <vector>
#include <cmath>
#include <iostream>
#include <functional>
#include <numeric>
#include <algorithm>
#include <random>
#include "cppctw.h"

using namespace std;

#define MAX_DEPTH 512

struct SuffixTreeNode {
    vector<SuffixTreeNode*> children;
    vector<int> counts;
    int tail_ind;
    char tail_symbol;

    static char alphabet_size;
    static double beta;

    double weighted_code_length;
    double local_code_length;


    SuffixTreeNode() {
        counts.assign(alphabet_size, 0);
        children.assign(alphabet_size, nullptr);
        tail_ind = -1;
        tail_symbol = -1;
    }

    SuffixTreeNode(int tail_ind, char tail_symbol) {
        counts.assign(alphabet_size, 0);
        children.assign(alphabet_size, nullptr);
        this->tail_ind = tail_ind;
        this->tail_symbol = tail_symbol;
    }

    ~SuffixTreeNode() { 
        vector<int>().swap(counts);
        for (char i = 0; i < alphabet_size; i++) {
            if ( this->children[i] != nullptr ) {
                delete this->children[i];
                this->children[i] = nullptr;
            }
        }
        vector<SuffixTreeNode*>().swap(children);
     }

    void update(char symbol) {
        counts[symbol]++;
    }

    void update_code_lengths() {
        // Assumes children have been updated already
        double sum_counts = accumulate(counts.begin(), counts.end(), 0.);
        double LE = lgamma(sum_counts + alphabet_size*beta) - lgamma(alphabet_size*beta);

        for (char i = 0; i < alphabet_size; i++) {
            LE -= lgamma(counts[i] + beta) - lgamma(beta);
        }
        LE /= log(2);
        local_code_length = LE;
        double L_C = 0;
        bool childfull = false;
        for (char i = 0; i < alphabet_size; i++) {
            if ( this->children[i] != nullptr ) {
                childfull = true;
                this->children[i]->update_code_lengths();
                L_C += this->children[i]->weighted_code_length;
            }
        }
        if (childfull && (sum_counts>1)) {
            weighted_code_length = 1 + min(L_C, local_code_length) - log2(1 + pow(2, -abs(local_code_length - L_C)));
        } else {
            weighted_code_length = local_code_length;
        }
    }

};

char SuffixTreeNode::alphabet_size = 0;
double SuffixTreeNode::beta = 0.;

class SuffixTree {
public:

    SuffixTree() {
        root = new SuffixTreeNode();

    }

    ~SuffixTree() {
        delete root;
    }
    
    float estimate_entropy() { 
        // updates the code lengths based on the counts
        root->update_code_lengths();
        return root->weighted_code_length / sequence_length;  
    };  
    
    void process_sequence(const vector<char>& sequence) { // fills out the tree
        sequence_length = sequence.size();
        SuffixTreeNode* currentNode = root;
        int depth;
        char context_symbol;
        char next_symbol_back_tail;
        char current_symbol;
        for (int current_position_ind = 0; current_position_ind < sequence_length; ++current_position_ind) {
            current_symbol = sequence[current_position_ind];
            // cout << current_symbol;

            currentNode = root;
            currentNode->update(current_symbol);
            if (current_position_ind > 0) {
                for (int context_position_ind = current_position_ind-1; context_position_ind >= 0; --context_position_ind) {
                    // First handle if the current node is a tail: add the next child with the old symbol
                    if (currentNode->tail_ind > 0) {
                        next_symbol_back_tail = sequence[currentNode->tail_ind-1];
                        currentNode->children[next_symbol_back_tail] = new SuffixTreeNode(
                            currentNode->tail_ind-1, currentNode->tail_symbol);
                        currentNode->children[next_symbol_back_tail]->update(currentNode->tail_symbol);
                        currentNode->tail_ind = -1;
                        currentNode->tail_symbol = -1;
                    }
                    // Now propagate as you would have
                    context_symbol = sequence[context_position_ind];
                    if (currentNode->children[context_symbol] == nullptr) {
                        depth = current_position_ind - context_position_ind;
                        if (depth > MAX_DEPTH) {
                            // cout << current_position_ind << "\t" << depth << "\n";
                            break;
                        }
                        // Make the next one a tail and stop
                        if (context_position_ind > 0) {
                            currentNode->children[context_symbol] = new SuffixTreeNode(
                                context_position_ind, current_symbol);
                        } else {
                            currentNode->children[context_symbol] = new SuffixTreeNode();
                        }
                        currentNode->children[context_symbol]->update(current_symbol);
                        break;
                    }
                    currentNode = currentNode->children[context_symbol];
                    currentNode->update(current_symbol);
                }

            }
        }
    }

private:
    int sequence_length;
    SuffixTreeNode* root;
};



double estimate_entropy(const vector<char>& sequence, char alphabet_size) {
    
    SuffixTreeNode::alphabet_size = alphabet_size;
    SuffixTreeNode::beta = 1. / alphabet_size;
    SuffixTree ctw = SuffixTree();
    ctw.process_sequence(sequence);
    return ctw.estimate_entropy();

}
